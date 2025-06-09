"""Utility methods and classes for turning the heterogeneous graph this project uses into PyTorch datasets to be trained
and tested on."""

from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_geometric.data import HeteroData

from models.graph import GraphEntityType, GraphRelationType

EntityRelationTripletDatasetOrSubset = TypeVar('EntityRelationTripletDatasetOrSubset', 
                                               'EntityRelationTripletDataset', 
                                               Subset['EntityRelationTripletDataset'])
"""Union of the `EntityRelationTripletDataset` class type and the `Subset` type returned by PyTorch's dataset splitting methods."""

RELATION_TYPE_TO_ID_MAP = {GraphRelationType.HAS_TRACK: 0, 
                           GraphRelationType.HAS_ARTIST: 1, 
                           GraphRelationType.IN_ALBUM: 2}
"""A mapping of the relation types in this project's heterogeneous graph to unique integers."""

class EntityRelationTripletDatasetEntryKey:
    """Keys for an `EntityRelationTripletDataset`'s entries, which are string-tensor dictionaries."""

    X = 'x'
    """The feature vector of whichever node of the triplet -- head or tail -- represents a track."""

    TRIPLET = 'triplet'
    """A three-element tensor `[head, tail, relation]` for an entity-relation-entity triplet, where `head` and `tail` 
    are the graph indices of this triplet's head and tail nodes, and `relation` is an ID for the type of relation 
    between the head and tail."""

class EntityRelationTripletDataset(Dataset):
    """
    A custom PyTorch dataset class for storing and accessing a heterogeneous graph's entity-relation-entity triplets. 
    Its entries are string-tensor dictionaries to be accessed by the keys in the `EntityRelationTripletDatasetEntryKey` 
    wrapper class.
    """

    x: torch.Tensor
    """The feature matrix for this graph's nodes representing tracks."""

    triplets: torch.Tensor
    """The graph's entity-relation-entity triplets in the form `[head, tail, relation]`, where `head` and `tail` are 
    the graph indices of a triplet's head and tail nodes, and `relation` is the ID to which the relation type between 
    head and tail is mapped by `RELATION_TYPE_TO_ID_MAP`."""

    def __init__(self, data: HeteroData):
        """
        Creates a new `EntityRelationTripletDataset` object from a PyTorch Geometric `HeteroData` object representing a
        heterogeneous graph of tracks, playlists, albums, artists, and the relations between them.
        #### Parameters:
        - `data`: the `HeteroData` object from which to assemble entity-relation-entity triplets
        """
        self.x = data[GraphEntityType.TRACK].x
        triplets = []
        for edge_type in data.metadata()[1]:
            # For each relation type, get its matrix of edge indices (i.e., the indices of each triplet's head and tail), 
            # transpose it, then add a new column for the ID that relation type maps to:
            edge_index_for_relation = data[edge_type].edge_index
            num_edges_for_relation = edge_index_for_relation.size(1)
            triplets_for_relation = torch.cat((edge_index_for_relation, 
                                               torch.full((1, num_edges_for_relation), RELATION_TYPE_TO_ID_MAP[edge_type[1]], dtype=torch.long)), 
                                               dim=0)
            # Shuffle the order in which the entity-relation-entity triplets are added:
            triplets.append(triplets_for_relation.T[torch.randperm(num_edges_for_relation)])
        self.triplets = torch.cat(triplets, dim=0)
    
    def __len__(self) -> int:
        return self.triplets.size(0)
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        head, tail, relation = triplet_at_index = self.triplets[index]
        return {EntityRelationTripletDatasetEntryKey.X: self.x[tail 
                                                               if (relation == RELATION_TYPE_TO_ID_MAP[GraphRelationType.HAS_TRACK]) 
                                                               else head],
                EntityRelationTripletDatasetEntryKey.TRIPLET: triplet_at_index}
    
    @staticmethod
    def split_by_fraction(dataset: EntityRelationTripletDatasetOrSubset, fraction: float) -> list[Subset['EntityRelationTripletDataset']]:
        """
        Randomly splits an input `EntityRelationTripletDataset`, or a subset of one, into two.
        #### Parameters:
        - `dataset`: the dataset to split
        - `fraction`: the fraction of dataset entries for one output data subset to take, and the other be left with
        #### Returns:
        two subsets (of PyTorch's `Subset` type) of the input dataset, its entries randomly apportioned among the two 
        according to `fraction`
        #### Throws:
        - `ValueError`: if `fraction` is not a single number between 0 and 1
        """
        if 0 <= fraction <= 1:
            return random_split(dataset, (fraction, 1 - fraction))
        raise ValueError('The fraction by which to split the input dataset must be a single number between 0 and 1.')
    
    @staticmethod
    def create_dataloader(dataset: EntityRelationTripletDatasetOrSubset, batch_size: int = 1024) -> DataLoader['EntityRelationTripletDataset']:
        """
        Wrapper method for creating a PyTorch `DataLoader` that loads entries of an `EntityRelationTripletDataset` (or a 
        subset of one) in shuffled batches.
        #### Parameters:
        - `dataset`: the dataset to return a `DataLoader` of
        - `batch_size`: the number of entries per data loader batch
        #### Returns:
        a PyTorch `DataLoader` object with parameters `dataset` and `batch_size` appropriately set, and `shuffle` and 
        `drop_last` set to `True`
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def create_hetero_data(x: torch.Tensor, 
                       playlist_to_track_edge_index: torch.Tensor,
                       track_to_artist_edge_index: torch.Tensor,
                       track_to_album_edge_index: torch.Tensor) -> HeteroData:
    """
    Creates a PyTorch Geometric `HeteroData` object for the heterogeneous graph of tracks, playlists, albums, artists,
    and the relations between them.
    #### Parameters:
    - `x`: a feature matrix for the graph's tracks, with dimensions `[num_track_nodes, num_track_features]`
    - `playlist_to_track_edge_index`: edge indices from playlist to track nodes indicating the 
    `GraphRelationType.HAS_TRACK` relation, with dimensions `[2, num_edges_from_playlists_to_tracks]`
    - `track_to_artist_edge_index`: edge indices from track to artist nodes indicating the 
    `GraphRelationType.HAS_ARTIST` relation, with dimensions `[2, num_edges_from_tracks_to_artists]`
    - `track_to_album_edge_index`: edge indices from track to album nodes indicating the 
    `GraphRelationType.IN_ALBUM` relation, with dimensions `[2, num_edges_from_tracks_to_albums]`
    #### Returns:
    a PyTorch Geometric `HeteroData` object with its node feature and edge index tensors defined accordingly
    #### Throws:
    - `ValueError`: if `num_edges_from_playlists_to_tracks` or `num_edges_from_tracks_to_artists` aren't greater than or
    equal to `num_track_nodes`, or if `num_edges_from_tracks_to_albums` isn't exactly equal to `num_track_nodes`. The
    former condition is held because each track in the graph must be in at least one playlist and have at least one 
    artist, and the latter because each track must be in only one album.
    """
    num_track_nodes = x.size(0)
    num_edges_from_playlists_to_tracks = playlist_to_track_edge_index.size(1)
    num_edges_from_tracks_to_artists = track_to_artist_edge_index.size(1)
    num_edges_from_tracks_to_albums = track_to_album_edge_index.size(1)
    if num_edges_from_playlists_to_tracks < num_track_nodes:
        raise ValueError(f"The dimension of playlist_to_track_edge_index ({num_edges_from_playlists_to_tracks}) should " 
                         + f"be greater than or equal to {num_track_nodes}, the dimension of the track node feature " 
                         + "matrix x.")
    if num_edges_from_tracks_to_artists < num_track_nodes:
        raise ValueError(f"The dimension of track_to_artist_edge_index ({num_edges_from_tracks_to_artists}) should be " 
                         + f"greater than or equal to {num_track_nodes}, the dimension of the track node feature matrix x.")
    if num_edges_from_tracks_to_albums != num_track_nodes:
        raise ValueError(f"The dimension of track_to_album_edge_index ({num_edges_from_tracks_to_albums}) should be " 
                         + f"equal to {num_track_nodes}, the dimension of the track node feature matrix x.")

    data = HeteroData()

    data[GraphEntityType.TRACK].x = x
    data[GraphEntityType.TRACK].num_nodes = num_track_nodes
    data[GraphEntityType.PLAYLIST].num_nodes = _get_num_nodes_from_edge_index_matrix(playlist_to_track_edge_index, from_first_row=True)
    data[GraphEntityType.ARTIST].num_nodes = _get_num_nodes_from_edge_index_matrix(track_to_artist_edge_index, from_first_row=False)
    data[GraphEntityType.ALBUM].num_nodes = _get_num_nodes_from_edge_index_matrix(track_to_album_edge_index, from_first_row=False)

    data[GraphEntityType.PLAYLIST, GraphRelationType.HAS_TRACK, GraphEntityType.TRACK].edge_index = playlist_to_track_edge_index
    data[GraphEntityType.TRACK, GraphRelationType.HAS_ARTIST, GraphEntityType.ARTIST].edge_index = track_to_artist_edge_index
    data[GraphEntityType.TRACK, GraphRelationType.IN_ALBUM, GraphEntityType.ALBUM].edge_index = track_to_album_edge_index
    return data

def _get_num_nodes_from_edge_index_matrix(edge_index: torch.Tensor, from_first_row: bool) -> float:
    return torch.max(edge_index, dim=1)[0][0 if from_first_row else 1].item()

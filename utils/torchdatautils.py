"""Utility methods and classes for turning the heterogeneous graph this project uses into PyTorch Geometric datasets to 
be trained and tested on."""

from typing import Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

from models.graph import GraphEntityType, GraphRelationType

RELATION_TYPE_TO_ID_MAP = {GraphRelationType.HAS_TRACK: 0, 
                           GraphRelationType.HAS_ARTIST: 1, 
                           GraphRelationType.IN_ALBUM: 2}
"""A mapping of the relation types in this project's heterogeneous graph to unique integers."""

RELATION_ID_TO_HEAD_TAIL_ENTITY_TYPES_MAP = {0: (GraphEntityType.PLAYLIST, GraphEntityType.TRACK), 
                                             1: (GraphEntityType.TRACK, GraphEntityType.ARTIST), 
                                             2: (GraphEntityType.TRACK, GraphEntityType.ALBUM)}
"""A mapping of integer IDs of the relations in this project's heterogeneous graph to the names of their associated head 
and tail entities."""

class EntityRelationTripletAndEgoNetworkDataLoaderKey:
    """Keys for the string-tensor dictionaries of batched data returned by `EntityRelationTripletAndEgoNetworkDataLoader`."""

    TRACK_FEATURE_MATRIX = 'track_feature_matrix'
    """The feature vectors of whichever nodes of a triplet batch -- for each triplet, either its head or its tail -- 
    represent tracks."""

    HTR_TRIPLET_BATCH = 'htr_triplet_batch'
    """A batch of three-element tensors `[head, tail, relation]` for entity-relation-entity triplets, where for each 
    triplet, `head` and `tail` are the graph indices of its head and tail nodes, and `relation` is an ID for the type 
    of relation between the head and tail."""

    UIJ_EGO_NETWORK_BATCH = 'uij_ego_network_batch'
    """A subgraph of edges of which the heads are within a number of hops of a batch of nodes representing playlists in
    the heterogeneous graph; i.e., these nodes' *ego-networks*, all the graph edges emanating from them."""
    

class EntityRelationTripletAndEgoNetworkDataLoader:
    """
    A wrapper class for loading batches of entity-relation-entity triplets and ego-network subgraphs rooted in 
    batches of playlist entity nodes. Following the KGAT training procedure outlined by Wang *et al.*, the triplets are 
    passed into the knowledge graph embedding layer of the model this project uses; the subgraphs into the graph 
    attention convolutional layers.
    """

    NUM_NEIGHBORS_PER_HOP: int = 10
    """The maximum number of neighbors to sample of each node when constructing the ego-network subgraphs."""

    htr_triplet_batch_loader: Iterator[list[torch.Tensor]]
    """An iterator wrapping a PyTorch `DataLoader`, which batches and returns entity-relation-entity triplets and track 
    feature vectors (stacked into a matrix)."""

    uij_ego_network_batch_loader: Iterator[HeteroData]
    """An iterator wrapping a PyTorch Geometric `LinkNeighborLoader`, which returns subgraphs expanded outward from 
    batches of playlist-to-track edges -- the ego-networks of the playlist nodes that head these edges. These batches 
    also have negative (false) playlist-to-track edges, to be used in model loss and backpropagation."""

    def __init__(self, data: HeteroData, batch_size: int = 1024, num_hops: int = 3):
        """
        Creates an `EntityRelationTripletAndEgoNetworkDataLoader` for a heterogeneous graph.
        #### Parameters: 
        - `data`: the heterogeneous graph, represented as a PyTorch Geometric `HeteroData` object
        - `batch_size`: the number of entity-relation-entity triplets and playlist-to-track edges per batch of data 
        returned by this loader
        - `num_hops`: the number of hops from each batch's playlist nodes to expand subgraphs (ego-networks) by
        """
        triplets = []
        track_feature_matrices = []
        for edge_type in data.edge_types:
            # For each relation type, get its matrix of edge indices (i.e., the indices of each triplet's head and tail), 
            # transpose it, then add a new column for the ID that relation type maps to:
            edge_index_for_relation = data[edge_type].edge_index
            num_edges_for_relation = edge_index_for_relation.size(1)
            triplets_for_relation = torch.cat((edge_index_for_relation, 
                                               torch.full((1, num_edges_for_relation), RELATION_TYPE_TO_ID_MAP[edge_type[1]], dtype=torch.long)), 
                                               dim=0)
            # Shuffle the order in which the entity-relation-entity triplets are added:
            shuffled_triplets = triplets_for_relation.T[torch.randperm(num_edges_for_relation)]
            triplets.append(shuffled_triplets)
            # Also retrieve the corresponding feature vectors for the nodes in the triplets representing tracks:
            track_feature_matrices.append(data[GraphEntityType.TRACK].x[
                shuffled_triplets[:, 0 if (edge_type[0] == GraphEntityType.TRACK) else 1]])
            
        # Wrap the triplets and track feature matrix in a PyTorch DataLoader:
        self.htr_triplet_batch_loader = iter(DataLoader(TensorDataset(*map(lambda x: torch.cat(x, dim=0), (triplets, track_feature_matrices))), 
                                                                 batch_size=batch_size, shuffle=True, drop_last=True))
   
        # Reverses the input graph's edges. PyTorch Geometric's LinkNeighborLoader, by default, builds subgraphs 
        # tail-to-head starting with a given edge type, but we want these subgraphs head-to-tail so that they are
        # ego-networks of batches of playlist entity nodes, and information flows "upstream" towards these nodes in message-passing.
        data_deepcopy = HeteroData()
        # Deep-copy all of the information stored in the input HeteroData object:
        for node_type in data.node_types:
            data_deepcopy[node_type].num_nodes = data[node_type].num_nodes
        for (head, relation, tail), edge_index in data.edge_index_dict.items():
            # Clone before flipping edge index tensors to avoid memory errors:
            flipped_edge_index = edge_index.clone().flip(0)
            data_deepcopy[(tail, relation, head)].edge_index = flipped_edge_index
        
        # Maintains a PyTorch Geometric LinkNeighborLoader to produce ego-network subgraphs rooted in playlist node batches,
        # also sampling negative playlist-to-track edges:
        u_i_j_edge_type = (GraphEntityType.TRACK, GraphRelationType.HAS_TRACK, GraphEntityType.PLAYLIST)
        self.uij_ego_network_batch_loader = iter(LinkNeighborLoader(data_deepcopy, 
                                                                      num_neighbors={e: [self.NUM_NEIGHBORS_PER_HOP] * (num_hops - 1) 
                                                                                     for e in data_deepcopy.edge_types}, 
                                                                      edge_label_index=(u_i_j_edge_type, data_deepcopy[u_i_j_edge_type].edge_index),
                                                                      edge_label=torch.ones(data_deepcopy[u_i_j_edge_type].edge_index.size(1)),
                                                                      neg_sampling_ratio=1, batch_size=batch_size, shuffle=True))
    
    def __iter__(self) -> 'EntityRelationTripletAndEgoNetworkDataLoader':
        """Returns this `EntityRelationTripletAndEgoNetworkDataLoader` instance, which itself implements the iterator 
        protocol.
        #### Returns: 
        this object itself
        """
        return self
    
    def __next__(self) -> dict[str, torch.Tensor | HeteroData]:
        """
        Produces a new batch of entity-relation-entity triplets, and a new subgraph of a batch of playlist entity 
        nodes' ego-networks.
        #### Returns:
        a dictionary with three entries: a batch of entity-relation-entity triplets (dimension `[batch_size, 3]`); a 
        matrix (`[batch_size, num_track_features]`), of which the rows are feature vectors for whichever node of the 
        triplet at the same index represents a track; and a `HeteroData` object for a `num_hops`-hop subgraph rooted 
        in `batch_size` playlist-to-track edges, also sampling an equal number of negative edges with the same playlist
        nodes as heads
        """
        htr_triplet_batch, track_feature_matrix = next(self.htr_triplet_batch_loader)
        return {
            EntityRelationTripletAndEgoNetworkDataLoaderKey.HTR_TRIPLET_BATCH: htr_triplet_batch,
            EntityRelationTripletAndEgoNetworkDataLoaderKey.TRACK_FEATURE_MATRIX: track_feature_matrix,
            EntityRelationTripletAndEgoNetworkDataLoaderKey.UIJ_EGO_NETWORK_BATCH: next(self.uij_ego_network_batch_loader)
        }

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
 
def train_test_split(data: HeteroData, fraction: float) -> tuple[HeteroData, HeteroData]:
    """
    Randomly splits an input `HeteroData` by its edges into two subgraphs intended for model training and testing.
    #### Parameters:
    - `data`: the `HeteroData` object the edges of which to split
    - `fraction`: the fraction of edges for one output subgraph to take, and the other be left with
    #### Returns:
    two subgraphs (themselves `HeteroData` instances) of the input, its edges randomly apportioned among the two 
    according to `fraction`
    #### Throws:
    - `ValueError`: if `fraction` is not a single number between 0 and 1
    """
    if (fraction < 0) or (fraction > 1):
        raise ValueError('The fraction by which to split the input HeteroData\'s edges must be a single number between 0 and 1.')
    traindata, _, testdata = RandomLinkSplit(num_val=0, num_test=(1 - fraction), is_undirected=False, split_labels=True,
                                             add_negative_train_samples=False, edge_types=data.edge_types)(data)
    return traindata, testdata

def _get_num_nodes_from_edge_index_matrix(edge_index: torch.Tensor, from_first_row: bool) -> float:
    return torch.max(edge_index, dim=1)[0][0 if from_first_row else 1].item() + 1
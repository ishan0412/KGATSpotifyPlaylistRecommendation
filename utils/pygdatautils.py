"""Utility methods for creating and manipulating PyTorch Geometric representations of the heterogeneous graph this 
project uses."""

from models.graph import GraphEntityType, GraphRelationType
import torch
from torch_geometric.data import HeteroData

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

    data[GraphEntityType.TRACK] = x
    data[GraphEntityType.PLAYLIST, GraphRelationType.HAS_TRACK, GraphEntityType.TRACK].edge_index = playlist_to_track_edge_index
    data[GraphEntityType.TRACK, GraphRelationType.HAS_ARTIST, GraphEntityType.ARTIST].edge_index = track_to_artist_edge_index
    data[GraphEntityType.TRACK, GraphRelationType.IN_ALBUM, GraphEntityType.ALBUM].edge_index = track_to_album_edge_index
    return data

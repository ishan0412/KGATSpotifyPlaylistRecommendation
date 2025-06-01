"""Utility methods for processing the Spotify playlist and track datasets into a graph the neural network can take as input."""

import os

import pandas as pd
import torch
from dotenv import load_dotenv

from constants import (
    GRAPH_FILE_NAME,
    PLAYLIST_DATASET_CSV_FILE_NAME,
    PLAYLIST_DATASET_ZIP_FILE_NAME,
    TRACK_DATASET_CSV_FILE_NAME,
    TRACK_DATASET_ZIP_FILE_NAME,
    TRACK_ID_LIST_FILE_NAME,
)
from models.datasetfields import PlaylistDatasetField, TrackDatasetField
from models.graph import GraphEntityType, GraphRelationType, GraphTripletField

load_dotenv()

def create_lists_of_trackids_and_entity_relation_triplets(n: int | None = None, random_state: int | None = None):
    """
    Creates a heterogeneous graph of "entities" (nodes: playlists, tracks, artists, and albums) and "relations" (edges)
    between them. This graph is kept in `graph.csv` as a list of entity-relation-entity triplets, having mapped each 
    entity to a unique numeric ID (its node index in the graph). A mapping of tracks' Spotify IDs (the `id` field in the 
    `tracks_features.csv` dataset) to their node indices is also maintained in `trackids.csv`.
    #### Parameters:
    - `n`: the number of playlist-track entries from the two datasets to use
    - `random_state`: the seed with which entries are randomly sampled
    #### Throws:
    - `FileNotFoundError`: if any of the dataset files haven't been downloaded yet or aren't in the data directory
    - `NameError`: if no path to a data directory has been set as an environment variable
    """
    # Make sure the playlist and track datasets have been downloaded. If not, raise an exception:
    playlist_dataset_csv = _get_path_to_dataset_csv(PLAYLIST_DATASET_ZIP_FILE_NAME, PLAYLIST_DATASET_CSV_FILE_NAME)
    track_dataset_csv = _get_path_to_dataset_csv(TRACK_DATASET_ZIP_FILE_NAME, TRACK_DATASET_CSV_FILE_NAME)
    path_to_graph_file = _get_path_to_processed_data_file(GRAPH_FILE_NAME)
    path_to_trackid_list_file = _get_path_to_processed_data_file(TRACK_ID_LIST_FILE_NAME)

    # Reads the datasets:
    trackdf = pd.read_csv(track_dataset_csv,
                          converters={TrackDatasetField.ARTISTS: lambda x: tuple(eval(x))})
    playlistdf = pd.read_csv(playlist_dataset_csv, on_bad_lines='skip').dropna(axis=0)
    # Process the playlist dataframe by making the comma-separated string entries for the "artist name" column into lists:
    playlistdf[PlaylistDatasetField.ARTIST_NAME] = playlistdf[PlaylistDatasetField.ARTIST_NAME].apply(
        _comma_and_ampersand_separated_string_to_tuple)

    # Merges the datasets by track and artist names (so that multiple tracks with the same name but different artists are 
    # distinguished):
    resultdf = pd.merge(trackdf, playlistdf, 
                        left_on=[TrackDatasetField.NAME, TrackDatasetField.ARTISTS], 
                        right_on=[PlaylistDatasetField.TRACK_NAME, PlaylistDatasetField.ARTIST_NAME],
                        how='inner').drop([PlaylistDatasetField.TRACK_NAME, PlaylistDatasetField.ARTIST_NAME], axis=1)
    # If n, the number of playlist-track entries to use, is given, choose those entries randomly:
    if n is not None:
        resultdf = resultdf.sample(n=n, random_state=random_state)
    # Assigns each track and user playlist their node indices (beginning at zero) in the graph:
    resultdf[TrackDatasetField.NAME], trackids = pd.factorize(resultdf[TrackDatasetField.ID])
    resultdf[PlaylistDatasetField.PLAYLIST_NAME] = (pd.factorize(resultdf[PlaylistDatasetField.USER_ID]
                                                                 .combine(resultdf[PlaylistDatasetField.PLAYLIST_NAME], 
                                                                          lambda x, y: x + y))[0])
    # Outputs a file for the mapping between the tracks' Spotify IDs and their graph node indices:
    trackids.to_series(index=range(len(trackids))).to_csv(path_to_trackid_list_file, index=True)
    print(f'Made a file for the mapping between {len(trackids)} tracks\' Spotify IDs and their graph node indices at '
          + f'{path_to_trackid_list_file}.')

    # Creates entities and relations (nodes and edges) between playlists and tracks, tracks and albums, and albums and artists:
    playlists_to_tracks_df = _get_entity_relation_tuple_dataframe(resultdf, 
                                                                  GraphEntityType.PLAYLIST, 
                                                                  GraphEntityType.TRACK, 
                                                                  GraphRelationType.HAS_TRACK)
    resultdf = resultdf.drop_duplicates(TrackDatasetField.NAME)
    tracks_to_albums_df = _get_entity_relation_tuple_dataframe(resultdf, 
                                                               GraphEntityType.TRACK, 
                                                               GraphEntityType.ALBUM, 
                                                               GraphRelationType.IN_ALBUM)
    tracks_to_artists_df = (resultdf[[TrackDatasetField.NAME, TrackDatasetField.ARTISTS]]
                            .explode(TrackDatasetField.ARTISTS))
    tracks_to_artists_df = _get_entity_relation_tuple_dataframe(tracks_to_artists_df, 
                                                                GraphEntityType.TRACK, 
                                                                GraphEntityType.ARTIST, 
                                                                GraphRelationType.HAS_ARTIST)
    # Outputs all of the entity-relation-entity triplets to a single file:
    pd.concat((playlists_to_tracks_df, tracks_to_albums_df, tracks_to_artists_df), 
              ignore_index=True).to_csv(path_to_graph_file, index=False)
    print(f'Made a file of entity-relation-entity triplets at {path_to_graph_file} defining a heterogenous graph with '
          + f'{len(playlists_to_tracks_df)} playlist-track relations, {len(tracks_to_albums_df)} track-album relations, '
          + f'and {len(tracks_to_artists_df)} track-artist relations.')

def create_feature_matrix_for_tracks(*features: str) -> torch.Tensor:
    """
    Creates a matrix of feature values for each track in both the playlist and track datasets.
    #### Parameters:
    - `features`: the fields of the track dataset to use as features
    #### Returns: 
    a PyTorch float tensor with dimensions `[num_track_nodes, num_track_features]`, where `num_track_nodes` is the 
    number of tracks in both datasets and `num_track_features = len(features)`. If a number of playlist-track entries to 
    use, `n`, was passed when processing the raw datasets into graphs with 
    `create_lists_of_trackids_and_entity_relation_triplets()`, then `num_track_nodes <= n`.
    #### Throws:
    - `FileNotFoundError`: if the track dataset hasn't been downloaded yet, or if `trackids.csv`, the file for the 
    mapping between tracks' Spotify IDs and graph node indices, hasn't been made yet or isn't in the data directory
    - `NameError`: if no path to a data directory has been set as an environment variable
    - `ValueError`: if any of `features` are not valid fields in the track dataset
    """
    if invalid_features := (set(features) - TrackDatasetField._FEATURE_SET):
        raise ValueError('Features "' + ','.join(invalid_features) + '" are not names of fields in the track dataset ' 
                         + 'and are therefore invalid. Use the constants in TrackDatasetField as input for this method.')
    trackdf = pd.read_csv(_get_path_to_dataset_csv(TRACK_DATASET_ZIP_FILE_NAME, TRACK_DATASET_CSV_FILE_NAME), 
                          usecols=features + (TrackDatasetField.ID,),
                          converters={TrackDatasetField.EXPLICIT: lambda x: int(eval(x))})
    path_to_trackid_list_file = _get_path_to_processed_data_file(TRACK_ID_LIST_FILE_NAME)
    if not os.path.isfile(path_to_trackid_list_file):
        raise FileNotFoundError('The file for the mapping between tracks\' Spotify IDs and graph node indices was not ' 
                                + f'found at {path_to_trackid_list_file}. If it hasn\'t been created yet, run '
                                + 'create_lists_of_trackids_and_entity_relation_triplets().')
    trackids = pd.read_csv(path_to_trackid_list_file, index_col=0)

    resultdf = pd.merge(trackids, trackdf, left_on=trackids.columns[0], right_on=TrackDatasetField.ID)[list(features)]
    return torch.tensor(resultdf.to_numpy(), dtype=torch.float)

def create_edge_index_matrix_for_relation_type(relationtype: str) -> torch.Tensor:
    """
    Creates a connectivity matrix for one of the three defined entity relations: `has_track` from playlists to tracks, 
    `has_artist` from tracks to artists, and `in_album` from tracks to albums.
    #### Parameters:
    - `relationtype`: the relation type
    #### Returns:
    a PyTorch long tensor with dimensions `[2, num_edges]`. This tensor's columns are pairs of graph node indices, of 
    which the corresponding entities are connected via `relationtype` relations. `num_edges` depends on `relationtype`:
    * if `GraphRelationType.HAS_TRACK` and a number `n` of playlist-track entries to use from the datasets was passed into
    `create_lists_of_trackids_and_entity_relation_triplets()`, then `num_edges = n` 
    * if `GraphRelationType.HAS_ARTIST`, then `num_edges >= num_track_nodes`, where `num_track_nodes` is the dimension 
    of the node feature matrix returned by `create_feature_matrix_for_tracks()`
    * if `GraphRelationType.IN_ALBUM`, then `num_edges = num_track_nodes`
    #### Throws:
    - `FileNotFoundError`: if `graph.csv` - the file defining the heterogenous graph of tracks, playlists, artists, 
        albums, and their relations - hasn't been made yet or isn't in the data directory
    - `NameError`: if no path to a data directory has been set as an environment variable
    - `ValueError`: if `relationtype` is not one of the members defined in `GraphRelationType`
    """
    if relationtype not in GraphRelationType._RELATIONS:
        raise ValueError(f'Input parameter "{relationtype}" must a member of GraphRelationType.')
    path_to_graph_file = _get_path_to_processed_data_file(GRAPH_FILE_NAME)
    if not os.path.isfile(path_to_graph_file):
        raise FileNotFoundError('The file defining the heterogenous graph of tracks, playlists, artists, and albums was ' 
                                + f'not found at {path_to_graph_file}. If this file hasn\'t been created yet, run '
                                + 'create_lists_of_trackids_and_entity_relation_triplets().')
    graphdf = pd.read_csv(path_to_graph_file)

    edge_index_df = graphdf.loc[graphdf[GraphTripletField.RELATION_TYPE] == relationtype, 
                                [GraphTripletField.HEAD_ID, GraphTripletField.TAIL_ID]]
    return torch.tensor(edge_index_df.to_numpy().T, dtype=torch.long)

# Private utility method for getting the path to any .csv dataset file in whichever folder data is kept.
def _get_path_to_dataset_csv(folder: str, filename: str) -> str:
    _raise_error_if_data_directory_not_set()
    path_to_folder = os.path.join(os.getenv('DATA_DIR'), folder) # type: ignore
    path_to_csv_file = os.path.join(path_to_folder, filename) + '.csv'
    if os.path.exists(path_to_folder) and os.path.isfile(path_to_csv_file):
        return path_to_csv_file
    raise FileNotFoundError(f'File {filename}.csv not found in {folder} folder. If the file hasn\'t been downloaded yet, '
                            f'run downloadutils.download_and_extract_url().')

# Private utility method for getting the path to any processed data file output by 
# create_lists_of_trackids_and_entity_relation_triplets().
def _get_path_to_processed_data_file(filename: str) -> str:
    _raise_error_if_data_directory_not_set()
    return os.path.join(os.getenv('DATA_DIR'), filename) + '.csv' # type: ignore

def _get_entity_relation_tuple_dataframe(df: pd.DataFrame, head: str, tail: str, relation: str) -> pd.DataFrame:
    hfield = _map_graph_entity_type_to_dataframe_field(head)
    tfield = _map_graph_entity_type_to_dataframe_field(tail)
    outputdf = (df[[hfield, tfield]]
                .rename({hfield: GraphTripletField.HEAD_ID, 
                         tfield: GraphTripletField.TAIL_ID}, axis=1))

    field_to_map_to_uids = (GraphTripletField.HEAD_ID if (head != GraphEntityType.TRACK) else GraphTripletField.TAIL_ID)
    outputdf[field_to_map_to_uids] = pd.factorize(outputdf[field_to_map_to_uids])[0]
    outputdf[GraphTripletField.HEAD_TYPE] = head
    outputdf[GraphTripletField.TAIL_TYPE] = tail
    outputdf[GraphTripletField.RELATION_TYPE] = relation
    return outputdf

def _map_graph_entity_type_to_dataframe_field(entitytype: str) -> str:
    match entitytype:
        case GraphEntityType.TRACK:
            return TrackDatasetField.NAME
        case GraphEntityType.PLAYLIST:
            return PlaylistDatasetField.PLAYLIST_NAME
        case GraphEntityType.ALBUM:
            return TrackDatasetField.ALBUM_ID
        case GraphEntityType.ARTIST:
            return TrackDatasetField.ARTISTS
    raise ValueError(f'"{entitytype}" is not a valid graph entity type. Use one of the constants in GraphEntityType.')

def _comma_and_ampersand_separated_string_to_tuple(s: str) -> tuple[str, ...]:
    s1 = s.split(',')
    return tuple(map(str.strip, s1[:-1] + s1[-1].split('&')))

def _raise_error_if_data_directory_not_set():
    if os.getenv('DATA_DIR') is None:
        raise NameError('No path to a data directory has been set as an environment variable.')
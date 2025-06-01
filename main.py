from constants import (
                                     PLAYLIST_DATASET_URL,
                                     PLAYLIST_DATASET_ZIP_FILE_NAME,
                                     TRACK_DATASET_URL,
                                     TRACK_DATASET_ZIP_FILE_NAME,
)
from models.datasetfields import TrackDatasetField
from models.graph import GraphRelationType
from utils.datautils import (
                                     create_edge_index_matrix_for_relation_type,
                                     create_feature_matrix_for_tracks,
                                     create_lists_of_trackids_and_entity_relation_triplets,
)
from utils.downloadutils import download_and_extract_url
from utils.pygdatautils import create_hetero_data

# Download the playlist and track datasets:
download_and_extract_url(PLAYLIST_DATASET_URL, PLAYLIST_DATASET_ZIP_FILE_NAME)
download_and_extract_url(TRACK_DATASET_URL, TRACK_DATASET_ZIP_FILE_NAME)

# Process the datasets into a heterogeneous graph kept in graph.csv:
create_lists_of_trackids_and_entity_relation_triplets(n=2000, random_state=42)

# Make a feature matrix for the graph nodes representing tracks:
x = create_feature_matrix_for_tracks(TrackDatasetField.DANCEABILITY,
                                     TrackDatasetField.ENERGY, 
                                     TrackDatasetField.EXPLICIT)
print(x[0])
print(x.shape)

# Make edge index matrices for each of the three relations defined between graph nodes/entities:
playlist_to_track_edge_index = create_edge_index_matrix_for_relation_type(GraphRelationType.HAS_TRACK)
track_to_artist_edge_index = create_edge_index_matrix_for_relation_type(GraphRelationType.HAS_ARTIST)
track_to_album_edge_index = create_edge_index_matrix_for_relation_type(GraphRelationType.IN_ALBUM)
print(playlist_to_track_edge_index.shape)
print(track_to_artist_edge_index.shape)
print(track_to_album_edge_index.shape)

print(create_hetero_data(x, playlist_to_track_edge_index, track_to_artist_edge_index, track_to_album_edge_index))


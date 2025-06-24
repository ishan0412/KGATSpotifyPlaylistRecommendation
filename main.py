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
from utils.torchdatautils import (
                                     EntityRelationTripletAndEgoNetworkDataLoader,
                                     create_hetero_data,
                                     train_test_split,
)

# Download the playlist and track datasets:
download_and_extract_url(PLAYLIST_DATASET_URL, PLAYLIST_DATASET_ZIP_FILE_NAME)
download_and_extract_url(TRACK_DATASET_URL, TRACK_DATASET_ZIP_FILE_NAME)

# Process the datasets into a heterogeneous graph kept in graph.csv:
create_lists_of_trackids_and_entity_relation_triplets(n=2000, random_state=42)

# Make a feature matrix for the graph nodes representing tracks:
x = create_feature_matrix_for_tracks(TrackDatasetField.DANCEABILITY,
                                     TrackDatasetField.ENERGY, 
                                     TrackDatasetField.EXPLICIT)
print('First row of track feature matrix: ', x[0])
print('Track feature matrix shape ([num_track_nodes, num_track_features]): ', x.shape, '\n')

# Make edge index matrices for each of the three relations defined between graph nodes/entities:
playlist_to_track_edge_index = create_edge_index_matrix_for_relation_type(GraphRelationType.HAS_TRACK)
track_to_artist_edge_index = create_edge_index_matrix_for_relation_type(GraphRelationType.HAS_ARTIST)
track_to_album_edge_index = create_edge_index_matrix_for_relation_type(GraphRelationType.IN_ALBUM)
print('Playlist-to-track edge index shape: ([2, num_edges_from_playlists_to_tracks]): ', playlist_to_track_edge_index.shape)
print('Track-to-artist edge index shape: ([2, num_edges_from_tracks_to_artists]): ', track_to_artist_edge_index.shape)
print('Track-to-album edge index shape: ([2, num_edges_from_tracks_to_albums]): ',track_to_album_edge_index.shape, '\n')

# Wrap the track feature matrix and edge index matrices into a PyTorch Geometric HeteroData object, then split it along 
# its edges into subgraphs to train and test on:
data = create_hetero_data(x, playlist_to_track_edge_index, track_to_artist_edge_index, track_to_album_edge_index)
trainset, testset = train_test_split(data, fraction=0.8)
print('HeteroData of edges to \033[1mtrain\033[0m on: ', trainset, '\n')
print('HeteroData of edges to \033[1mtest\033[0m on: ', testset, '\n')

# Put the training subgraph into a custom data loader that batches both entity-relation-entity triplets and playlist 
# node-headed subgraphs to train on:
trainloader = EntityRelationTripletAndEgoNetworkDataLoader(trainset, batch_size=512)
print(next(iter(trainloader)))

"""Field (column) names of the playlist and track datasets this project uses."""

class PlaylistDatasetField:
    """Field names of the playlist dataset, `spotify_dataset.csv`."""

    PLAYLIST_NAME = ' "playlistname"'
    """The name of the playlist that contains this track."""

    TRACK_NAME = ' "trackname"'
    """The title of a track."""

    ARTIST_NAME = ' "artistname"'
    """The name of the artist(s)."""

    USER_ID = 'user_id'
    """A hash of the user's Spotify user name."""

class TrackDatasetField:
    """Field names of the track dataset, `tracks_features.csv`."""

    NAME = 'name'
    """A track's title."""

    ID = 'id'
    """A track's Spotify ID."""

    ALBUM_ID = 'album_id'
    """The Spotify ID of the album a track belongs to."""

    ARTIST_IDS = 'artist_ids'
    """A list of Spotify IDs of the artists on a track."""

    ARTISTS = 'artists'
    """A list of names of the artists on a track."""

    EXPLICIT = 'explicit'
    """Whether the song is explicit (1) or not (0)."""

    DANCEABILITY = 'danceability'
    """How suitable a track is for dancing, scored between 0 and 1."""

    ENERGY = 'energy'
    """How intense and active a track is, scored between 0 and 1."""

    LOUDNESS = 'loudness'
    """Overall loudness of a track in decibels."""

    MODE = 'mode'
    """Whether a track is in major mode (1) or minor (0)."""

    SPEECHINESS = 'speechiness'
    """Proportion of spoken words in a track, scored between 0 and 1."""

    ACOUSTICNESS = 'acousticness'
    """Confidence measure of whether a track is acoustic, scored between 0 and 1."""

    INSTRUMENTALNESS = 'instrumentalness'
    """Proportion of instrumental parts in a track, scored between 0 and 1."""

    VALENCE = 'valence'
    """Measures how positive a track sounds, from 1 (extremely positive) to 0 (extremely negative)"""

    TEMPO = 'tempo'
    """Overall tempo of a track in beats per minute (BPM)."""

    _FEATURE_SET = {'explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 
                    'instrumentalness', 'valence', 'tempo'}
    """Fields that can be used as training features (i.e., passed into `create_feature_matrix_for_tracks()`)."""
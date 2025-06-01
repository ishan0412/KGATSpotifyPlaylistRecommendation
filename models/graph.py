"""Classes of constants referencing the heterogeneous graph on which this project's neural network will learn, 
consisting of entities (playlists, tracks, artists, and albums) and relations between them."""

class GraphEntityType:
    """The different entities (nodes) in the graph."""

    PLAYLIST = 'playlist'
    TRACK = 'track'
    ARTIST = 'artist'
    ALBUM = 'album'

class GraphRelationType:
    """The different relations (edges) between entities in the graph."""

    HAS_TRACK = 'has_track'
    HAS_ARTIST = 'has_artist'
    IN_ALBUM = 'in_album'
    _RELATIONS = {'has_track', 'has_artist', 'in_album'}

class GraphTripletField:
    """Field names of `graph.csv`, which stores the graph as a list of entity-relation-entity triplets."""

    HEAD_TYPE = 'headtype'
    """The head entity/source node's type."""

    HEAD_ID = 'headid'
    """The head entity/source node's index."""

    TAIL_TYPE = 'tailtype'
    """The tail entity/destination node's type."""

    TAIL_ID = 'tailid'
    """The tail entity/destination node's index."""

    RELATION_TYPE = 'relationtype'
    """The relation (edge type) between the head and tail entities."""
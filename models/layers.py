"""Implementations of the layers in Wang et al.'s KGAT model."""

import torch
import torch.nn as nn

from utils.torchdatautils import (
    RELATION_TYPE_TO_ID_MAP,
    EntityRelationTripletDatasetEntryKey,
)

from .graph import GraphEntityType, GraphRelationType


class KnowledgeGraphEmbeddingLayerOutputKey:
    """Keys for each entry in the output of `KnowledgeGraphEmbeddingLayer`'s forward pass."""

    HEAD_EMBEDDINGS = 'head_embeddings'
    TAIL_EMBEDDINGS = 'tail_embeddings'
    RELATION_EMBEDDINGS = 'relation_embeddings'
    SCORES = 'scores'

class KnowledgeGraphEmbeddingLayer(nn.Module):
    """
    An embedding layer for learning vector representations of the entities and relations in this project's heterogeneous 
    graph, scoring and optimizing these embeddings using the TransR method.
    """

    entity_embeddings: nn.Embedding
    """A lookup table of the embeddings of all the graph's entities, partitioned by entity type - track, playlist, album,
    or artist."""

    relation_embeddings: nn.Embedding
    """A lookup table of embeddings for each relation type in the graph."""

    relation_projection_matrices: nn.Embedding
    """Matrices for each relation type that project entity embeddings into relation embedding vector space."""

    track_feature_and_embedding_projection_layer: nn.Linear
    """A linear neural network layer for tracks, taking their concatenated feature vectors and embeddings as input."""

    type_offsets: dict[str, int]
    """A dictionary for referencing at which index in `entity_embeddings` each entity type's embeddings begin."""

    embedding_dim: int
    """Dimension of both entity and relation embeddings."""

    def __init__(self, num_nodes_by_type: dict[str, int], track_feature_dim: int, embedding_dim: int = 64):
        """
        Creates a new `KnowledgeGraphEmbeddingLayer` with all entity and relation embeddings initialized.
        #### Parameters:
        - `num_nodes_by_type`: a dictionary, of which each entry is an entity type and how many entities (nodes) of that
        type there are in the heterogeneous graph
        - `track_feature_dim`: the number of features being used for tracks
        - `embedding_dim`: the dimension of this layer's entity and relation embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        # Create learnable embeddings for each entity in the graph and keeps them all in a single nn.Embedding layer, in
        # which each entity type is allocated its own contiguous range:
        self.type_offsets = {}
        num_total_nodes = 0
        for entity_type, num_nodes in num_nodes_by_type.items():
            self.type_offsets[entity_type] = num_total_nodes
            num_total_nodes += num_nodes
        self.entity_embeddings = nn.Embedding(num_total_nodes, embedding_dim)
        # Learnable embeddings for each relation type in the graph:
        num_relation_types = len(GraphRelationType._RELATIONS)
        self.relation_embeddings = nn.Embedding(num_relation_types, embedding_dim)
        # Learnable entity-to-relation projection matrices for each relation type to be used in TransR:
        self.relation_projection_matrices = nn.Embedding(num_relation_types, embedding_dim * embedding_dim)
        # Create a simple linear layer for nodes representing tracks, projecting their concatenated feature and 
        # embedding vectors to embedding_dim, the same dimensionality as all the other nodes and relations' embeddings:
        self.track_feature_and_embedding_projection_layer = nn.Linear(track_feature_dim + embedding_dim, embedding_dim)

        # Xavier initializer for all model parameters:
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        nn.init.xavier_normal_(self.relation_projection_matrices.weight)
        nn.init.xavier_normal_(self.track_feature_and_embedding_projection_layer.weight)
    
    def _project(self, entity_embedding: torch.Tensor, relation_id: torch.Tensor) -> torch.Tensor:
        m = self.relation_projection_matrices(relation_id).view(-1, self.embedding_dim, self.embedding_dim)
        return torch.bmm(entity_embedding.unsqueeze(1), m.transpose(1, 2)).squeeze(1)
        
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Performs this layer's forward pass on a heterogeneous graph represented as entity-relation-entity triplets and
        feature vectors for track nodes (entities).
        #### Parameters:
        - `input`: a dictionary with two entries: one of entity-relation-entity triplets, and the other a feature matrix,
        the rows of which correspond to tracks
        #### Returns:
        a dictionary with four entries: the embeddings of the triplets' heads, tails, and relations, and the triplets'
        TransR scores
        """
        x = input[EntityRelationTripletDatasetEntryKey.X]
        triplets = input[EntityRelationTripletDatasetEntryKey.TRIPLET]
        head_embeddings = torch.empty((triplets.size(0), self.embedding_dim))
        tail_embeddings = torch.empty((triplets.size(0), self.embedding_dim))
        for relation in GraphRelationType._RELATIONS:
            # For each defined relation, use only the triplets of that relation type:
            mask = triplets[:, 2] == RELATION_TYPE_TO_ID_MAP[relation]
            x_for_relation, triplets_for_relation = x[mask], triplets[mask]
            # Determine the types of the triplets' head and tail entities:
            match relation:
                case GraphRelationType.HAS_TRACK:
                    head_entity_type = GraphEntityType.PLAYLIST
                    tail_entity_type = GraphEntityType.TRACK
                case GraphRelationType.HAS_ARTIST:
                    head_entity_type = GraphEntityType.TRACK
                    tail_entity_type = GraphEntityType.ARTIST
                case GraphRelationType.IN_ALBUM:
                    head_entity_type = GraphEntityType.TRACK
                    tail_entity_type = GraphEntityType.ALBUM
                case _:
                    break
            # Use the heads and tails' types (and their corresponding type offsets) to look up their embeddings:
            head_embeddings_for_relation = self.entity_embeddings(self.type_offsets[head_entity_type] + triplets_for_relation[:, 0])
            tail_embeddings_for_relation = self.entity_embeddings(self.type_offsets[tail_entity_type] + triplets_for_relation[:, 1])
            # Determine which, heads or tails, are the tracks, and pass their embeddings and feature vectors, concatenated, 
            # into the linear projection layer created at instantiation:
            if head_entity_type == GraphEntityType.TRACK:
                head_embeddings_for_relation = self.track_feature_and_embedding_projection_layer(
                    torch.concat((head_embeddings_for_relation, x_for_relation), dim=1))
            else:
                tail_embeddings_for_relation = self.track_feature_and_embedding_projection_layer(
                    torch.concat((tail_embeddings_for_relation, x_for_relation), dim=1))
                
            indices = torch.nonzero(mask).squeeze()
            head_embeddings[indices] = head_embeddings_for_relation
            tail_embeddings[indices] = tail_embeddings_for_relation
            
        # Apply the projection matrices onto the entity embeddings:
        projected_head_embeddings = self._project(head_embeddings, triplets[:, 2])
        projected_tail_embeddings = self._project(tail_embeddings, triplets[:, 2])
        # Compute TransR scores between relation embeddings and head and tail entity embeddings projected into relation space:
        relation_embeddings = self.relation_embeddings(triplets[:, 2])
        scores = torch.linalg.vector_norm(projected_head_embeddings + relation_embeddings - projected_tail_embeddings, dim=1)
        return {
            KnowledgeGraphEmbeddingLayerOutputKey.HEAD_EMBEDDINGS: head_embeddings,
            KnowledgeGraphEmbeddingLayerOutputKey.TAIL_EMBEDDINGS: tail_embeddings,
            KnowledgeGraphEmbeddingLayerOutputKey.RELATION_EMBEDDINGS: relation_embeddings,
            KnowledgeGraphEmbeddingLayerOutputKey.SCORES: scores
        }

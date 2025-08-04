"""
GraphRAG Entity Models for Educational Knowledge Graph

This module defines the enhanced entity and relationship models
for building a comprehensive knowledge graph from exam questions.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class EntityType(str, Enum):
    """Types of entities in the educational knowledge graph"""
    QUESTION = "Question"
    TOPIC = "Topic"
    CONCEPT = "Concept"
    EXAM_PAPER = "ExamPaper"
    SKILL = "Skill"
    LEARNING_OBJECTIVE = "LearningObjective"
    LEARNER_PROFILE = "LearnerProfile"
    DIFFICULTY_LEVEL = "DifficultyLevel"
    BLOOM_TAXONOMY = "BloomTaxonomy"
    SUBJECT = "Subject"
    CURRICULUM = "Curriculum"
    ASSESSMENT_TYPE = "AssessmentType"


class RelationType(str, Enum):
    """Types of relationships between entities"""
    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"
    
    # Hierarchical relationships
    BELONGS_TO = "BELONGS_TO"
    PART_OF = "PART_OF"
    SUBSUMES = "SUBSUMES"
    
    # Educational relationships
    REQUIRES = "REQUIRES"  # Prerequisites
    TESTS = "TESTS"  # What a question tests
    HAS_DIFFICULTY = "HAS_DIFFICULTY"
    HAS_COGNITIVE_LEVEL = "HAS_COGNITIVE_LEVEL"
    
    # Learning relationships
    PEDAGOGICAL_SEQUENCE = "PEDAGOGICAL_SEQUENCE"
    CONCEPT_DEPENDENCY = "CONCEPT_DEPENDENCY"
    COMPOSED_OF = "COMPOSED_OF"
    
    # Assessment relationships
    APPEARS_IN = "APPEARS_IN"
    ASSESSES = "ASSESSES"


class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels"""
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


class DifficultyScale(str, Enum):
    """Difficulty levels for questions"""
    VERY_EASY = "Very Easy"
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    VERY_HARD = "Very Hard"


class GraphEntity(BaseModel):
    """Base entity model for knowledge graph"""
    id: str = Field(..., description="Unique identifier for the entity")
    type: EntityType = Field(..., description="Type of the entity")
    name: str = Field(..., description="Display name of the entity")
    description: Optional[str] = Field(None, description="Detailed description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class GraphRelationship(BaseModel):
    """Relationship between entities in the knowledge graph"""
    id: Optional[str] = Field(None, description="Unique identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: RelationType = Field(..., description="Type of relationship")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Strength of relationship")
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class GraphQuestion(BaseModel):
    """Enhanced question model with graph properties"""
    id: str
    question_number: str
    question_text: str
    marks: int = Field(..., ge=0)
    topics: List[str] = Field(..., description="Main topics covered")
    concepts: List[str] = Field(default_factory=list, description="Key concepts tested")
    skills: List[str] = Field(default_factory=list, description="Skills required")
    difficulty_score: float = Field(0.5, ge=0.0, le=1.0, description="Normalized difficulty")
    difficulty_level: DifficultyScale = Field(DifficultyScale.MEDIUM)
    bloom_levels: List[BloomLevel] = Field(default_factory=list, description="Cognitive levels")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisite concepts")
    year: str
    level: str
    source_pdf: str
    question_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphCommunity(BaseModel):
    """Community (cluster) of related entities"""
    id: str
    name: str
    summary: str = Field(..., description="Natural language summary of the community")
    member_ids: List[str] = Field(..., description="IDs of entities in this community")
    centroid_embedding: Optional[List[float]] = Field(None, description="Community centroid")
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResult(BaseModel):
    """Result of entity and relationship extraction"""
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from GraphRAG search"""
    entity: GraphEntity
    score: float = Field(..., ge=0.0, le=1.0)
    context: List[GraphEntity] = Field(default_factory=list, description="Related entities")
    relationships: List[GraphRelationship] = Field(default_factory=list)
    community: Optional[GraphCommunity] = None
    search_type: str = Field(..., description="local, global, or hybrid")


class LearningPath(BaseModel):
    """Learning path through concepts"""
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = Field(..., description="Ordered learning steps")
    prerequisites: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    estimated_duration: Optional[int] = Field(None, description="Duration in minutes")
    difficulty_progression: List[DifficultyScale] = Field(default_factory=list)
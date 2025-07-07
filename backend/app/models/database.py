# app/models/database.py - Day 2 Team Workspace Models
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, ForeignKey, JSON, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timezone
from typing import Optional

Base = declarative_base()

class Workspace(Base):
    """Team workspace for collaborative compliance work"""
    __tablename__ = "workspaces"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    organization = Column(String(255))
    country = Column(String(2), default="DE")  # ISO country code
    industry = Column(String(100))  # automotive, healthcare, manufacturing
    compliance_frameworks = Column(JSON, default=["gdpr"])
    
    # German enterprise specific
    german_authority = Column(String(100))  # BfDI, BayLDA, etc.
    dpo_contact = Column(String(255))  # Data Protection Officer
    legal_entity_type = Column(String(50), default="GmbH")
    
    # Workspace settings
    language_preference = Column(String(5), default="de")
    timezone = Column(String(50), default="Europe/Berlin")
    gdpr_consent = Column(Boolean, default=False)
    audit_level = Column(String(20), default="standard")  # basic, standard, enhanced
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Subscription info (for Day 4 SME pricing)
    subscription_tier = Column(String(20), default="sme")  # sme, enterprise
    max_documents = Column(Integer, default=100)
    max_users = Column(Integer, default=10)
    
    # Relationships
    users = relationship("User", back_populates="workspace")
    documents = relationship("Document", back_populates="workspace")
    analyses = relationship("ComplianceAnalysis", back_populates="workspace")

class User(Base):
    """Users within team workspaces"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"))
    
    # German enterprise roles
    role = Column(String(50), default="analyst")  # admin, dpo, analyst, viewer
    german_certification = Column(String(100))  # TÜV certified DPO, etc.
    language_preference = Column(String(5), default="de")
    
    # Authentication (simplified for Day 2)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime)
    
    # GDPR compliance
    gdpr_consent_date = Column(DateTime)
    data_processing_consent = Column(Boolean, default=False)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="users")
    uploaded_documents = relationship("Document", foreign_keys="Document.uploaded_by")

class Document(Base):
    """Documents uploaded for compliance analysis"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))  # Store original name
    
    # File metadata
    file_size = Column(Integer)  # bytes
    file_type = Column(String(50))  # pdf, docx, txt, md
    mime_type = Column(String(100))
    content_hash = Column(String(64))  # SHA-256 for deduplication
    
    # Processing metadata
    language_detected = Column(String(5))
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    docling_metadata = Column(JSON)  # Docling processing results
    
    # German compliance metadata
    german_document_type = Column(String(100))  # Datenschutzerklärung, Verfahrensverzeichnis, etc.
    compliance_category = Column(String(50))  # policy, procedure, assessment, etc.
    dsgvo_relevance_score = Column(Float)
    
    # Upload info
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime)
    
    # GDPR compliance
    gdpr_deleted_at = Column(DateTime)  # When content was securely deleted
    retention_until = Column(DateTime)  # When document should be deleted
    
    # Relationships
    workspace = relationship("Workspace", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    analyses = relationship("ComplianceAnalysis", back_populates="document")

class DocumentChunk(Base):
    """Intelligent document chunks from Docling processing"""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(50))  # paragraph, table, list, header, footer
    page_number = Column(Integer)
    position_in_page = Column(Integer)
    
    # Content (temporarily stored, deleted after processing for GDPR compliance)
    content = Column(Text)
    content_hash = Column(String(64))
    char_count = Column(Integer)
    
    # Docling intelligence
    confidence_score = Column(Float)
    structural_importance = Column(Float)  # How important this chunk is structurally
    
    # German compliance analysis
    language_detected = Column(String(5))
    german_terms = Column(JSON)  # Detected German legal terms
    dsgvo_articles = Column(JSON)  # Referenced GDPR articles
    compliance_tags = Column(JSON)  # Compliance-relevant tags
    
    # Processing timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    analyzed_at = Column(DateTime)
    deleted_at = Column(DateTime)  # GDPR compliance - when content was deleted
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class ComplianceAnalysis(Base):
    """Compliance analysis results for documents"""
    __tablename__ = "compliance_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)  # Nullable for batch analyses
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Analysis metadata
    framework = Column(String(50), nullable=False)  # gdpr, soc2, hipaa, iso27001
    analysis_type = Column(String(50), default="document")  # document, batch, workspace
    
    # Results
    analysis_results = Column(JSON)  # Structured analysis results
    compliance_score = Column(Float)
    confidence_level = Column(Float)
    
    # German-specific analysis
    german_language_detected = Column(Boolean, default=False)
    dsgvo_compliance_score = Column(Float)
    german_authority_compliance = Column(JSON)  # BfDI, LfDI compliance checks
    
    # Processing metadata
    chunk_count = Column(Integer, default=0)
    processing_time_seconds = Column(Float)
    ai_model_used = Column(String(50))
    docling_version = Column(String(20))
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="analyses")
    document = relationship("Document", back_populates="analyses")

class AuditLog(Base):
    """GDPR-compliant audit trail for all system actions"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # System actions may not have user
    
    # Action details
    action = Column(String(100), nullable=False)  # document_uploaded, analysis_started, etc.
    resource_type = Column(String(50))  # document, analysis, user, workspace
    resource_id = Column(UUID(as_uuid=True))
    
    # Context
    details = Column(JSON)  # Additional context and metadata
    ip_address = Column(String(45))  # IPv4/IPv6
    user_agent = Column(Text)
    
    # German compliance specific
    gdpr_basis = Column(String(100))  # Legal basis for processing (Art. 6)
    data_category = Column(String(50))  # personal_data, compliance_data, etc.
    
    # Status and result
    status = Column(String(20), default="success")  # success, failure, pending
    error_message = Column(Text)
    
    # Timestamp
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Retention (for GDPR compliance)
    retain_until = Column(DateTime)  # When this audit log should be deleted

class ComplianceTemplate(Base):
    """German industry-specific compliance templates (Day 3-4 feature prep)"""
    __tablename__ = "compliance_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    industry = Column(String(100))  # automotive, healthcare, manufacturing
    framework = Column(String(50))  # gdpr, iso27001, etc.
    
    # German specific
    german_authority = Column(String(100))  # Which authority this template addresses
    legal_requirements = Column(JSON)  # Required documents and controls
    
    # Template content
    checklist_items = Column(JSON)
    required_documents = Column(JSON)
    compliance_controls = Column(JSON)
    
    # Metadata
    created_by = Column(String(100), default="WolfMerge")
    language = Column(String(5), default="de")
    version = Column(String(10), default="1.0")
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
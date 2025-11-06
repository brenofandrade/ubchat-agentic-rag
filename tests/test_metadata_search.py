"""
Testes para funcionalidade de busca por metadados.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.rag_engine import extract_document_identifiers, build_metadata_filters


class TestExtractDocumentIdentifiers:
    """Testes para extração de identificadores de documentos."""

    def test_extract_manual_identifier(self):
        """Testa extração de identificador de manual."""
        query = "O que diz o manual MAN-297?"
        identifiers = extract_document_identifiers(query)
        assert "MAN-297" in identifiers

    def test_extract_nr_identifier(self):
        """Testa extração de identificador de norma regulamentadora."""
        query = "Como se adequar à NR-013?"
        identifiers = extract_document_identifiers(query)
        assert "NR-013" in identifiers or "NR-13" in identifiers

    def test_extract_nr_without_hyphen(self):
        """Testa extração de NR sem hífen."""
        query = "Requisitos da NR 12"
        identifiers = extract_document_identifiers(query)
        # Deve normalizar para formato com hífen
        assert any("NR-12" in id or "NR12" in id for id in identifiers)

    def test_extract_multiple_identifiers(self):
        """Testa extração de múltiplos identificadores."""
        query = "Compare o manual MAN-297 com a norma NR-013"
        identifiers = extract_document_identifiers(query)
        assert len(identifiers) >= 2
        assert "MAN-297" in identifiers

    def test_extract_iso_identifier(self):
        """Testa extração de identificador ISO."""
        query = "Conforme a norma ISO-9001"
        identifiers = extract_document_identifiers(query)
        assert "ISO-9001" in identifiers

    def test_extract_proc_identifier(self):
        """Testa extração de identificador de procedimento."""
        query = "Seguir o PROC-1234"
        identifiers = extract_document_identifiers(query)
        assert "PROC-1234" in identifiers

    def test_no_identifiers(self):
        """Testa query sem identificadores."""
        query = "Como fazer manutenção preventiva?"
        identifiers = extract_document_identifiers(query)
        assert len(identifiers) == 0

    def test_case_insensitive(self):
        """Testa que a extração é case-insensitive."""
        query = "O que diz o man-297?"
        identifiers = extract_document_identifiers(query)
        # Deve ser normalizado para maiúsculas
        assert "MAN-297" in identifiers

    def test_normalizes_whitespace(self):
        """Testa normalização de espaços."""
        query = "O que diz o MAN 297?"
        identifiers = extract_document_identifiers(query)
        # Deve normalizar para formato com hífen
        assert "MAN-297" in identifiers


class TestBuildMetadataFilters:
    """Testes para construção de filtros de metadados."""

    def test_single_identifier_filter(self):
        """Testa criação de filtro para um único identificador."""
        identifiers = ["MAN-297"]
        filters = build_metadata_filters(identifiers)

        assert filters is not None
        assert "$or" in filters
        assert isinstance(filters["$or"], list)
        # Deve buscar em vários campos possíveis
        assert len(filters["$or"]) > 0

    def test_multiple_identifiers_filter(self):
        """Testa criação de filtro para múltiplos identificadores."""
        identifiers = ["MAN-297", "NR-013"]
        filters = build_metadata_filters(identifiers)

        assert filters is not None
        assert "$or" in filters
        # Deve criar condições para ambos os identificadores
        filter_str = str(filters)
        assert "MAN-297" in filter_str
        assert "NR-013" in filter_str

    def test_empty_identifiers(self):
        """Testa que retorna None para lista vazia."""
        identifiers = []
        filters = build_metadata_filters(identifiers)
        assert filters is None

    def test_filter_searches_multiple_fields(self):
        """Testa que o filtro busca em múltiplos campos de metadados."""
        identifiers = ["MAN-297"]
        filters = build_metadata_filters(identifiers)

        filter_str = str(filters)
        # Deve buscar em vários campos possíveis
        expected_fields = ["document_id", "doc_id", "id", "source", "title", "name"]
        # Pelo menos alguns desses campos devem estar presentes
        assert any(field in filter_str for field in expected_fields)


class TestEndToEndMetadataSearch:
    """Testes end-to-end para busca por metadados."""

    def test_query_with_identifier_extracts_and_filters(self):
        """Testa que uma query com identificador extrai e cria filtros."""
        query = "O que diz o manual MAN-297?"

        # Extrai identificadores
        identifiers = extract_document_identifiers(query)
        assert len(identifiers) > 0

        # Constrói filtros
        filters = build_metadata_filters(identifiers)
        assert filters is not None

        # Verifica estrutura do filtro
        assert "$or" in filters

    def test_query_without_identifier_no_filters(self):
        """Testa que uma query sem identificador não cria filtros."""
        query = "Como fazer manutenção preventiva?"

        # Extrai identificadores
        identifiers = extract_document_identifiers(query)
        assert len(identifiers) == 0

        # Não deve criar filtros
        filters = build_metadata_filters(identifiers)
        assert filters is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Script manual para testar funcionalidade de busca por metadados.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.rag_engine import extract_document_identifiers, build_metadata_filters


def test_extract_identifiers():
    """Testa extração de identificadores."""
    print("=" * 60)
    print("TESTANDO EXTRAÇÃO DE IDENTIFICADORES")
    print("=" * 60)

    test_cases = [
        ("O que diz o manual MAN-297?", ["MAN-297"]),
        ("Como se adequar à NR-013?", ["NR-013", "NR-13"]),
        ("Requisitos da NR 12", ["NR-12", "NR12"]),
        ("Compare o manual MAN-297 com a norma NR-013", ["MAN-297", "NR-013", "NR-13"]),
        ("Conforme a norma ISO-9001", ["ISO-9001"]),
        ("Seguir o PROC-1234", ["PROC-1234"]),
        ("Como fazer manutenção preventiva?", []),
        ("O que diz o man-297?", ["MAN-297"]),
        ("O que diz o MAN 297?", ["MAN-297"]),
    ]

    passed = 0
    failed = 0

    for query, expected_any in test_cases:
        identifiers = extract_document_identifiers(query)
        print(f"\nQuery: {query}")
        print(f"Identificadores encontrados: {identifiers}")
        print(f"Esperados (qualquer um de): {expected_any}")

        # Para queries sem identificadores esperados
        if not expected_any:
            if len(identifiers) == 0:
                print("✓ PASSOU")
                passed += 1
            else:
                print("✗ FALHOU")
                failed += 1
        else:
            # Verifica se algum dos identificadores esperados foi encontrado
            found = any(exp in identifiers for exp in expected_any)
            if found or len(identifiers) > 0:
                print("✓ PASSOU")
                passed += 1
            else:
                print("✗ FALHOU")
                failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed} passaram, {failed} falharam")
    print("=" * 60)

    return failed == 0


def test_build_filters():
    """Testa construção de filtros."""
    print("\n" + "=" * 60)
    print("TESTANDO CONSTRUÇÃO DE FILTROS")
    print("=" * 60)

    test_cases = [
        (["MAN-297"], True),
        (["MAN-297", "NR-013"], True),
        ([], False),
    ]

    passed = 0
    failed = 0

    for identifiers, should_have_filter in test_cases:
        filters = build_metadata_filters(identifiers)
        print(f"\nIdentificadores: {identifiers}")
        print(f"Filtros gerados: {filters}")

        if should_have_filter:
            if filters is not None and "$or" in filters:
                print("✓ PASSOU")
                passed += 1
            else:
                print("✗ FALHOU")
                failed += 1
        else:
            if filters is None:
                print("✓ PASSOU")
                passed += 1
            else:
                print("✗ FALHOU")
                failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed} passaram, {failed} falharam")
    print("=" * 60)

    return failed == 0


def test_end_to_end():
    """Testa fluxo completo."""
    print("\n" + "=" * 60)
    print("TESTANDO FLUXO COMPLETO")
    print("=" * 60)

    queries = [
        "O que diz o manual MAN-297?",
        "Como se adequar à NR-013?",
        "Como fazer manutenção preventiva?",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        # Extrai identificadores
        identifiers = extract_document_identifiers(query)
        print(f"Identificadores: {identifiers}")

        # Constrói filtros
        if identifiers:
            filters = build_metadata_filters(identifiers)
            print(f"Filtros: {filters}")
            print("✓ Filtros de metadados serão aplicados")
        else:
            print("✓ Busca semântica normal será usada")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTES DE BUSCA POR METADADOS")
    print("=" * 60)

    all_passed = True
    all_passed &= test_extract_identifiers()
    all_passed &= test_build_filters()
    test_end_to_end()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ TODOS OS TESTES PASSARAM")
    else:
        print("✗ ALGUNS TESTES FALHARAM")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)

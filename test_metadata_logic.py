"""
Teste isolado da lógica de extração de identificadores.
"""

import re
from typing import List, Optional, Dict, Any


def extract_document_identifiers(query: str) -> List[str]:
    """
    Extrai identificadores de documentos da query.
    (Cópia da função implementada para teste isolado)
    """
    patterns = [
        r'\b([A-Z]{2,6}-\d{2,6})\b',
        r'\b([A-Z]{2,6}\s*\d{2,6})\b',
        r'\b(NR\s*-?\s*\d{1,3})\b',
    ]

    identifiers = []
    for pattern in patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            identifier = match.group(1).strip()
            identifier = re.sub(r'\s+', '', identifier)
            identifier = re.sub(r'([A-Z]+)(\d+)', r'\1-\2', identifier, flags=re.IGNORECASE)
            identifier = identifier.upper()
            if identifier not in identifiers:
                identifiers.append(identifier)

    return identifiers


def build_metadata_filters(identifiers: List[str]) -> Optional[Dict[str, Any]]:
    """
    Constrói filtros de metadados do Pinecone baseado em identificadores.
    (Cópia da função implementada para teste isolado)
    """
    if not identifiers:
        return None

    if len(identifiers) == 1:
        identifier = identifiers[0]
        return {
            "$or": [
                {"document_id": {"$eq": identifier}},
                {"doc_id": {"$eq": identifier}},
                {"id": {"$eq": identifier}},
                {"source": {"$eq": identifier}},
                {"title": {"$eq": identifier}},
                {"name": {"$eq": identifier}},
                {"document_id": {"$in": [identifier]}},
                {"source": {"$in": [identifier]}},
            ]
        }
    else:
        or_conditions = []
        for identifier in identifiers:
            or_conditions.extend([
                {"document_id": {"$eq": identifier}},
                {"doc_id": {"$eq": identifier}},
                {"id": {"$eq": identifier}},
                {"source": {"$eq": identifier}},
                {"title": {"$eq": identifier}},
                {"name": {"$eq": identifier}},
            ])
        return {"$or": or_conditions}


def run_tests():
    """Executa todos os testes."""
    print("=" * 70)
    print("TESTES DE BUSCA POR METADADOS")
    print("=" * 70)

    # Teste 1: Extração de identificadores
    print("\n1. TESTANDO EXTRAÇÃO DE IDENTIFICADORES")
    print("-" * 70)

    test_cases = [
        ("O que diz o manual MAN-297?", "MAN-297"),
        ("Como se adequar à NR-013?", "NR-013"),
        ("Requisitos da NR 12", "NR-12"),
        ("Compare o manual MAN-297 com a norma NR-013", 2),  # 2 identificadores
        ("Conforme a norma ISO-9001", "ISO-9001"),
        ("Seguir o PROC-1234", "PROC-1234"),
        ("Como fazer manutenção preventiva?", None),  # Nenhum identificador
        ("O que diz o man-297?", "MAN-297"),  # Case insensitive
        ("O que diz o MAN 297?", "MAN-297"),  # Com espaço
    ]

    passed = 0
    failed = 0

    for query, expected in test_cases:
        identifiers = extract_document_identifiers(query)
        print(f"\nQuery: '{query}'")
        print(f"Identificadores: {identifiers}")

        if expected is None:
            if len(identifiers) == 0:
                print("✓ PASSOU - Nenhum identificador encontrado (esperado)")
                passed += 1
            else:
                print("✗ FALHOU - Esperava nenhum identificador")
                failed += 1
        elif isinstance(expected, int):
            if len(identifiers) == expected:
                print(f"✓ PASSOU - {expected} identificadores encontrados")
                passed += 1
            else:
                print(f"✗ FALHOU - Esperava {expected} identificadores, encontrou {len(identifiers)}")
                failed += 1
        else:
            if expected in identifiers:
                print(f"✓ PASSOU - '{expected}' encontrado")
                passed += 1
            else:
                print(f"✗ FALHOU - Esperava '{expected}'")
                failed += 1

    print(f"\n{'=' * 70}")
    print(f"Extração: {passed}/{passed+failed} testes passaram")

    # Teste 2: Construção de filtros
    print(f"\n2. TESTANDO CONSTRUÇÃO DE FILTROS")
    print("-" * 70)

    passed2 = 0
    failed2 = 0

    # Caso 1: Filtro para um identificador
    filters = build_metadata_filters(["MAN-297"])
    print("\nIdentificadores: ['MAN-297']")
    print(f"Filtro gerado: {filters is not None and '$or' in filters}")
    if filters and "$or" in filters and len(filters["$or"]) > 0:
        print("✓ PASSOU - Filtro criado com sucesso")
        print(f"  Campos buscados: {len(filters['$or'])} condições")
        passed2 += 1
    else:
        print("✗ FALHOU - Filtro não criado corretamente")
        failed2 += 1

    # Caso 2: Filtro para múltiplos identificadores
    filters = build_metadata_filters(["MAN-297", "NR-013"])
    print("\nIdentificadores: ['MAN-297', 'NR-013']")
    if filters and "$or" in filters:
        filter_str = str(filters)
        has_both = "MAN-297" in filter_str and "NR-013" in filter_str
        print(f"Filtro contém ambos identificadores: {has_both}")
        if has_both:
            print("✓ PASSOU - Filtro criado com ambos identificadores")
            passed2 += 1
        else:
            print("✗ FALHOU - Filtro não contém ambos identificadores")
            failed2 += 1
    else:
        print("✗ FALHOU - Filtro não criado")
        failed2 += 1

    # Caso 3: Sem identificadores
    filters = build_metadata_filters([])
    print("\nIdentificadores: []")
    if filters is None:
        print("✓ PASSOU - Nenhum filtro criado (esperado)")
        passed2 += 1
    else:
        print("✗ FALHOU - Não deveria criar filtro")
        failed2 += 1

    print(f"\n{'=' * 70}")
    print(f"Filtros: {passed2}/{passed2+failed2} testes passaram")

    # Teste 3: Fluxo completo
    print(f"\n3. TESTANDO FLUXO COMPLETO (END-TO-END)")
    print("-" * 70)

    examples = [
        "O que diz o manual MAN-297?",
        "Como se adequar à NR-013?",
        "Como fazer manutenção preventiva?",
    ]

    for query in examples:
        print(f"\n{'-' * 70}")
        print(f"Query: '{query}'")
        identifiers = extract_document_identifiers(query)
        print(f"Identificadores detectados: {identifiers}")

        if identifiers:
            filters = build_metadata_filters(identifiers)
            print("✓ Busca por metadados será aplicada")
            print(f"  Filtro: {len(filters['$or'])} condições no filtro $or")
        else:
            print("✓ Busca semântica normal será usada (sem filtros)")

    # Resultado final
    total_passed = passed + passed2
    total_tests = passed + failed + passed2 + failed2

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"Total: {total_passed}/{total_tests} testes passaram")

    if failed == 0 and failed2 == 0:
        print("✓ TODOS OS TESTES PASSARAM!")
        return True
    else:
        print("✗ ALGUNS TESTES FALHARAM")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

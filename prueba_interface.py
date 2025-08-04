#!/usr/bin/env python3
"""
Script de prueba para la interfaz Vector Store
Prototipo_chatbot - TFM Vicente Caruncho
"""

import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_interface():
    """Prueba básica de la interfaz Vector Store"""
    print("🚀 Iniciando prueba de Vector Store Interface...")
    print("=" * 60)
    
    try:
        print("📦 1. Importando módulos...")
        
        # Test de imports básicos
        from app.services.rag.vector_store import (
            VectorStoreInterface, 
            VectorStoreManager, 
            BenchmarkMetrics,
            SearchResult
        )
        print("   ✅ Interfaces importadas correctamente")
        
        # Test de funciones de conveniencia
        from app.services.rag.vector_store import (
            vector_store_manager,
            get_vector_store_benchmarks
        )
        print("   ✅ Funciones de conveniencia importadas")
        
    except ImportError as e:
        print(f"   ❌ Error importando: {e}")
        print("   💡 Verifica que los archivos estén en las rutas correctas")
        return False
    
    try:
        print("\n🔧 2. Creando instancias...")
        
        # Crear manager
        manager = VectorStoreManager()
        print(f"   ✅ VectorStoreManager creado: {type(manager)}")
        
        # Crear métricas de ejemplo
        metrics = BenchmarkMetrics(
            operation="test",
            vector_store_type="test_store", 
            execution_time=0.1
        )
        print(f"   ✅ BenchmarkMetrics creado: {metrics.operation}")
        
    except Exception as e:
        print(f"   ❌ Error creando instancias: {e}")
        return False
    
    try:
        print("\n⚙️ 3. Probando funcionalidades básicas...")
        
        # Test del manager vacío
        stores_count = len(manager.stores)
        print(f"   ✅ Manager inicializado con {stores_count} stores")
        
        # Test de benchmarks vacíos
        benchmarks = manager.get_all_benchmarks()
        print(f"   ✅ Benchmarks obtenidos: {len(benchmarks)} stores")
        
        # Test de reporte vacío
        report = manager.get_comparison_report()
        print(f"   ✅ Reporte generado: {len(report.get('stores_compared', []))} stores")
        
    except Exception as e:
        print(f"   ❌ Error en funcionalidades: {e}")
        return False
    
    try:
        print("\n📊 4. Probando funciones globales...")
        
        # Test función global de benchmarks
        global_benchmarks = get_vector_store_benchmarks()
        print(f"   ✅ Benchmarks globales: {type(global_benchmarks)}")
        
        # Test de instancia global
        global_manager = vector_store_manager
        print(f"   ✅ Manager global: {type(global_manager)}")
        
    except Exception as e:
        print(f"   ❌ Error en funciones globales: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
    print("✅ La interfaz Vector Store está funcionando correctamente")
    print("🚀 Lista para implementar FAISS y ChromaDB")
    print("=" * 60)
    
    return True

def test_mock_implementation():
    """Prueba con una implementación mock"""
    print("\n🧪 Prueba adicional: Implementación Mock")
    print("-" * 40)
    
    try:
        from app.services.rag.vector_store import VectorStoreInterface
        
        # Crear implementación mock para pruebas
        class MockVectorStore(VectorStoreInterface):
            def get_store_type(self):
                return "mock"
            
            def initialize(self):
                self.is_initialized = True
                return True
            
            def add_documents(self, chunks):
                return True
            
            def search(self, query_embedding, k=5, threshold=0.7):
                return []
            
            def delete_document(self, document_id):
                return True
            
            def update_document(self, chunk):
                return True
            
            def get_document_count(self):
                return 0
            
            def get_memory_usage(self):
                return 10.5  # MB simulados
        
        print("   ✅ Implementación Mock creada")
        
        # Crear y registrar mock store
        mock_store = MockVectorStore({})
        print(f"   ✅ Mock store instanciado: {mock_store.get_store_type()}")
        
        # Registrar en manager
        from app.services.rag.vector_store import vector_store_manager
        vector_store_manager.register_store(mock_store, is_default=True)
        print("   ✅ Mock store registrado en manager")
        
        # Inicializar
        init_results = vector_store_manager.initialize_all()
        print(f"   ✅ Inicialización: {init_results}")
        
        # Test de obtener store
        retrieved_store = vector_store_manager.get_store("mock")
        print(f"   ✅ Store recuperado: {retrieved_store.get_store_type()}")
        
        print("\n🎉 ¡Implementación Mock funciona perfectamente!")
        print("✅ La interfaz está lista para implementaciones reales")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en implementación mock: {e}")
        import traceback
        print(f"   📄 Traceback: {traceback.format_exc()}")
        return False

def main():
    """Función principal de pruebas"""
    print("🤖 Prototipo_chatbot - Test Vector Store Interface")
    print(f"📁 Directorio de trabajo: {os.getcwd()}")
    print(f"🐍 Python: {sys.version}")
    print()
    
    # Prueba básica de interfaz
    success_basic = test_interface()
    
    if success_basic:
        # Prueba con implementación mock
        success_mock = test_mock_implementation()
        
        if success_mock:
            print("\n🏆 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
            print("🔥 ¡Listo para implementar FAISS y ChromaDB!")
        else:
            print("\n⚠️ Prueba mock falló, pero interfaz básica funciona")
    else:
        print("\n❌ Pruebas básicas fallaron")
        print("💡 Revisa los imports y la estructura de archivos")
    
    print("\n📋 Próximos pasos:")
    print("1. Implementar FAISSVectorStore")
    print("2. Implementar ChromaDBVectorStore") 
    print("3. Crear servicio de embeddings")
    print("4. Integración completa RAG")

if __name__ == "__main__":
    main()
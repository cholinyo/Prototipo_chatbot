#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal MEJORADO
Chatbot RAG para Administraciones Locales con Pipeline Integrado
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""
import sys
import os
import time
import traceback
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_missing_files():
    """Crear archivos de configuración faltantes"""
    print("📁 Verificando estructura de directorios...")

    # Crear config/settings.yaml si no existe
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)

    settings_file = config_dir / "settings.yaml"
    if not settings_file.exists():
        print("⚠️ Creando archivo de configuración faltante...")

    # Crear directorios faltantes
    required_dirs = [
        "logs",
        "data/vectorstore/faiss",
        "data/vectorstore/chromadb",
        "data/documents",  # Directorio para documentos a ingestar
        "data/reports",    # Para reportes de benchmarking
        "app/static/css",
        "app/static/js",
        "app/templates/errors",
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

    print("✅ Estructura de directorios verificada")


def get_llm_service():
    """Obtener instancia del servicio LLM con manejo de errores"""
    try:
        from app.services.llm.llm_services import LLMService
        return LLMService()
    except ImportError as e:
        raise ImportError(f"No se pudo importar LLMService: {e}")
    except Exception as e:
        raise Exception(f"Error inicializando LLMService: {e}")


def get_rag_pipeline():
    """Obtener instancia del pipeline RAG con manejo de errores MEJORADO"""
    try:
        # Intentar importar el pipeline RAG mejorado
        from app.services.rag.pipeline import get_rag_pipeline
        return get_rag_pipeline()
    except ImportError as e:
        try:
            # Fallback a pipeline existente
            from app.services.rag_pipeline import get_rag_pipeline
            return get_rag_pipeline()
        except ImportError as e2:
            raise ImportError(f"No se pudo importar Pipeline RAG: {e}, {e2}")
    except Exception as e:
        raise Exception(f"Error inicializando Pipeline RAG: {e}")


def get_memory_usage():
    """Obtener uso de memoria del proceso"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return round(memory_mb, 2)
    except ImportError:
        return 0


def create_error_response(error_type, message, components_status=None):
    """Crear respuesta de error estandarizada MEJORADA"""
    base_response = {
        "status": "error",
        "timestamp": time.time(),
        "error_type": error_type,
        "error": message,
        "services": {
            "flask": "available",
            "ollama": "error",
            "openai": "error",
        },
        "components": components_status
        or {
            "embeddings": "error",
            "vector_store": "error",
            "llm": "error",
            "pipeline": "error",
        },
    }
    return jsonify(base_response)


def create_fallback_template(app_config):
    """Crear template HTML básico si no existe el principal MEJORADO"""
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <title>{app_config.name}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .card-custom {{ backdrop-filter: blur(10px); background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); }}
            .pulse {{ animation: pulse 2s infinite; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
        </style>
    </head>
    <body class="gradient-bg">
        <div class="container">
            <div class="row justify-content-center align-items-center min-vh-100">
                <div class="col-lg-8">
                    <div class="card card-custom text-white">
                        <div class="card-body text-center p-5">
                            <div class="mb-4">
                                <i class="fas fa-robot fa-4x pulse"></i>
                            </div>
                            <h1 class="display-4 mb-3">{app_config.name}</h1>
                            <p class="lead mb-4">{app_config.description}</p>
                            <p class="text-white-50 mb-4">TFM Vicente Caruncho - Sistemas Inteligentes UJI</p>

                            <div class="row g-3 mb-4">
                                <div class="col-md-6">
                                    <a href="/dashboard" class="btn btn-primary btn-lg w-100">
                                        <i class="fas fa-chart-line me-2"></i>Dashboard
                                    </a>
                                </div>
                                <div class="col-md-6">
                                    <a href="/health" class="btn btn-outline-light btn-lg w-100">
                                        <i class="fas fa-heartbeat me-2"></i>Estado Sistema
                                    </a>
                                </div>
                            </div>

                            <div class="row g-3">
                                <div class="col-md-4">
                                    <a href="/chat" class="btn btn-success w-100">
                                        <i class="fas fa-comments me-2"></i>Chat RAG
                                    </a>
                                </div>
                                <div class="col-md-4">
                                    <a href="/docs" class="btn btn-info w-100">
                                        <i class="fas fa-book me-2"></i>Documentación
                                    </a>
                                </div>
                                <div class="col-md-4">
                                    <a href="/about" class="btn btn-secondary w-100">
                                        <i class="fas fa-info-circle me-2"></i>Sobre TFM
                                    </a>
                                </div>
                            </div>

                            <div class="mt-4 pt-3 border-top border-white-50">
                                <small class="text-white-50">
                                    <i class="fas fa-code me-1"></i>Versión {getattr(app_config, 'version', '1.0.0')} |
                                    <i class="fas fa-github me-1"></i>GitHub: cholinyo/Prototipo_chatbot
                                </small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """


def register_blueprints(app):
    """Registrar blueprints de la aplicación MEJORADO"""
    logger.info("Registrando blueprints...")

    # Lista de blueprints a intentar registrar (orden de prioridad)
    blueprints_config = [
        ("app.routes.main", "main_bp", "/"),
        ("app.routes.chat_routes", "chat_bp", "/"),  # Chat en raíz para acceso directo
        ("app.routes.api", "api_bp", "/api"),
    ]

    registered_count = 0
    registration_details = []

    for module_path, blueprint_name, url_prefix in blueprints_config:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)

            # Obtener información de rutas del blueprint
            route_count = len(
                [rule for rule in app.url_map.iter_rules() if rule.endpoint.startswith(blueprint_name.replace("_bp", ""))]
            )

            registration_details.append(
                {
                    "name": blueprint_name,
                    "prefix": url_prefix,
                    "routes": route_count,
                    "status": "success",
                }
            )

            logger.info(f"✅ Blueprint '{blueprint_name}' registrado en {url_prefix} ({route_count} rutas)")
            registered_count += 1

        except ImportError as e:
            registration_details.append(
                {"name": blueprint_name, "prefix": url_prefix, "error": str(e), "status": "import_error"}
            )
            logger.warning(f"⚠️ Blueprint '{blueprint_name}' no disponible: {e}")

        except AttributeError as e:
            registration_details.append(
                {"name": blueprint_name, "prefix": url_prefix, "error": str(e), "status": "attribute_error"}
            )
            logger.error(f"❌ Error en blueprint '{blueprint_name}': {e}")

    logger.info(f"📋 {registered_count}/{len(blueprints_config)} blueprints registrados exitosamente")

    # Guardar detalles de registro para diagnóstico
    app.blueprint_registration = registration_details

    return registered_count


def create_flask_app():
    """Crear y configurar la aplicación Flask MEJORADO"""
    print("🔧 Configurando aplicación Flask...")

    # Intentar importar configuración
    try:
        from app.core.config import get_app_config, is_development  # noqa: F401
        from app.core.logger import setup_logging, get_logger

        setup_logging()
        logger = get_logger("main")
        app_config = get_app_config()
        logger.info("✅ Configuración importada correctamente")

    except ImportError as e:
        print(f"⚠️ Configuración no disponible, usando valores por defecto: {e}")

        # Configuración por defecto mejorada
        class DefaultConfig:
            name = "Prototipo_chatbot"
            version = "1.2.0"  # Versión actualizada
            description = "Sistema RAG Integrado para Administraciones Locales"
            host = "localhost"
            port = 5000
            debug = True
            environment = "development"

        app_config = DefaultConfig()

        # Logger básico mejorado
        class SimpleLogger:
            def info(self, msg): print(f"ℹ️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
            def debug(self, msg):
                if app_config.debug:
                    print(f"🔍 {msg}")

        logger = SimpleLogger()

    # Crear aplicación Flask con configuración mejorada
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static",
        instance_relative_config=True,
    )

    # Configuración básica de Flask mejorada
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
        DEBUG=app_config.debug,
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=app_config.debug,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
        UPLOAD_FOLDER=project_root / "data" / "uploads",
        WTF_CSRF_ENABLED=not app_config.debug,  # CSRF solo en producción
    )

    # Crear directorio de uploads si no existe
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    return app, app_config, logger


def setup_routes(app, app_config, logger):
    """Configurar rutas de la aplicación MEJORADO"""
    logger.info("Configurando rutas principales...")

    @app.route("/")
    def index():
        """Página principal MEJORADA"""
        try:
            return render_template(
                "index.html",
                app_name=app_config.name,
                app_version=getattr(app_config, "version", "1.2.0"),
                app_description=app_config.description,
            )
        except Exception as e:
            logger.warning(f"Template index.html no encontrado, usando fallback: {e}")
            return create_fallback_template(app_config)

    @app.route("/health")
    def health_check():
        """Endpoint de verificación de salud principal MEJORADO"""
        try:
            # Verificar LLM Service
            llm_service = get_llm_service()
            llm_health = llm_service.health_check()

            # NUEVA FUNCIONALIDAD: Verificar Pipeline RAG
            pipeline_health = {"status": "unavailable", "components": {}}
            try:
                rag_pipeline = get_rag_pipeline()
                if rag_pipeline and hasattr(rag_pipeline, "health_check"):
                    pipeline_health = rag_pipeline.health_check()
                    pipeline_available = True
                else:
                    pipeline_available = False
            except Exception as pipeline_error:
                logger.debug(f"Pipeline RAG no disponible: {pipeline_error}")
                pipeline_available = False

            # Respuesta combinada
            response = {
                "status": "healthy"
                if (llm_health["status"] == "healthy" and pipeline_health["status"] in ["healthy", "unavailable"])
                else "partial",
                "timestamp": llm_health["timestamp"],
                "services": {
                    "flask": "available",
                    "ollama": llm_health["services"]["ollama"]["status"],
                    "openai": llm_health["services"]["openai"]["status"],
                    "pipeline_rag": pipeline_health["status"],
                },
                "models": llm_health.get("models", {}),
                "components": {
                    "embeddings": "available" if pipeline_available else "limited",
                    "vector_store": pipeline_health["components"].get("vector_store", "unavailable"),
                    "llm": llm_health["status"],
                    "pipeline": pipeline_health["status"],
                },
                "system_info": {
                    "pipeline_available": pipeline_available,
                    "memory_usage_mb": get_memory_usage(),
                    "uptime_seconds": time.time() - app.start_time if hasattr(app, "start_time") else 0,
                },
            }

            logger.debug(f"Health check: {response['status']} (Pipeline: {pipeline_available})")
            return jsonify(response)

        except ImportError as e:
            logger.error(f"Error importando servicios: {e}")
            return create_error_response("import_error", str(e)), 500
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return create_error_response("health_error", str(e)), 500

    @app.route("/api/status")
    def api_status():
        """Endpoint de estado detallado para el dashboard MEJORADO"""
        try:
            # Obtener estados de servicios
            llm_service = get_llm_service()
            llm_health = llm_service.health_check()

            # Estado del pipeline RAG
            pipeline_info = {"available": False, "stats": {}}
            try:
                rag_pipeline = get_rag_pipeline()
                if rag_pipeline:
                    pipeline_health = rag_pipeline.health_check()
                    pipeline_stats = rag_pipeline.get_stats()

                    pipeline_info = {
                        "available": True,
                        "status": pipeline_health["status"],
                        "components": pipeline_health["components"],
                        "stats": pipeline_stats,
                    }
            except Exception as e:
                logger.debug(f"Pipeline RAG no disponible para status: {e}")

            # Respuesta detallada
            response = {
                "status": llm_health["status"],
                "timestamp": llm_health["timestamp"],
                "services": {
                    "flask": {
                        "status": "healthy",
                        "url": f"http://{app_config.host}:{app_config.port}",
                        "uptime": time.time() - app.start_time if hasattr(app, "start_time") else 0,
                        "version": getattr(app_config, "version", "1.2.0"),
                        "environment": getattr(app_config, "environment", "development"),
                    },
                    "ollama": llm_health["services"]["ollama"],
                    "openai": llm_health["services"]["openai"],
                    "pipeline_rag": {
                        "status": pipeline_info.get("status", "unavailable"),
                        "available": pipeline_info["available"],
                        "components": pipeline_info.get("components", {}),
                        "documents_count": pipeline_info.get("stats", {}).get("documents_count", 0),
                        "vector_store_type": pipeline_info.get("stats", {}).get("vector_store_type", "Unknown"),
                    },
                    "embeddings": {
                        "status": "healthy" if pipeline_info["available"] else "limited",
                        "model": pipeline_info.get("stats", {}).get("embedding_model", "all-MiniLM-L6-v2"),
                        "dimensions": pipeline_info.get("stats", {}).get("embedding_dimensions", 384),
                    },
                },
                "models": llm_health.get("models", {}),
                "system_info": {
                    "uptime_hours": round((time.time() - app.start_time) / 3600, 2) if hasattr(app, "start_time") else 0,
                    "version": getattr(app_config, "version", "1.2.0"),
                    "environment": "development" if app_config.debug else "production",
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "memory_usage_mb": get_memory_usage(),
                    "blueprint_count": len(getattr(app, "blueprint_registration", [])),
                    "pipeline_available": pipeline_info["available"],
                },
                "performance": {
                    "total_queries": pipeline_info.get("stats", {}).get("total_queries", 0),
                    "avg_response_time": pipeline_info.get("stats", {}).get("avg_response_time", 0),
                    "success_rate": pipeline_info.get("stats", {}).get("success_rate", 0),
                    "vector_store_size_mb": pipeline_info.get("stats", {}).get("vector_store_size_mb", 0),
                },
            }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error en api_status: {e}")
            return create_error_response("status_error", str(e)), 500

    @app.route("/ajax/quick-stats")
    def ajax_quick_stats():
        """Estadísticas rápidas para actualización automática MEJORADAS"""
        try:
            # Estadísticas base
            uptime_hours = round((time.time() - app.start_time) / 3600, 2) if hasattr(app, "start_time") else 0

            stats = {
                "queries": 0,
                "documents": 0,
                "avg_response_time": 0,
                "success_rate": 0,
                "uptime": uptime_hours,
                "ollama_models": 0,
                "openai_available": False,
                "system_status": "unknown",
                "last_update": time.time(),
                "memory_usage": get_memory_usage(),
                "active_connections": 1,
                "pipeline_available": False,
            }

            # Enriquecer con datos del pipeline RAG si está disponible
            try:
                rag_pipeline = get_rag_pipeline()
                if rag_pipeline:
                    pipeline_stats = rag_pipeline.get_stats()

                    stats.update(
                        {
                            "queries": pipeline_stats.get("total_queries", 0),
                            "documents": pipeline_stats.get("documents_count", 0),
                            "avg_response_time": round(pipeline_stats.get("avg_response_time", 0), 2),
                            "success_rate": round(pipeline_stats.get("success_rate", 0), 1),
                            "pipeline_available": True,
                            "vector_store_type": pipeline_stats.get("vector_store_type", "Unknown"),
                            "queries_today": pipeline_stats.get("queries_today", 0),
                        }
                    )
            except Exception as e:
                logger.debug(f"Pipeline no disponible para quick stats: {e}")

            # Datos de LLM
            try:
                llm_service = get_llm_service()
                llm_health = llm_service.health_check()

                stats.update(
                    {
                        "ollama_models": len(llm_health.get("models", {}).get("ollama", [])),
                        "openai_available": llm_health["services"]["openai"]["status"] == "configured",
                        "system_status": llm_health["status"],
                    }
                )
            except Exception as e:
                logger.debug(f"LLM Service no disponible para quick stats: {e}")
                stats["system_status"] = "error"

            return jsonify(stats)

        except Exception as e:
            logger.error(f"Error en quick_stats: {e}")
            return (
                jsonify(
                    {
                        "queries": 0,
                        "documents": 0,
                        "avg_response_time": 0,
                        "success_rate": 0,
                        "uptime": 0,
                        "ollama_models": 0,
                        "openai_available": False,
                        "system_status": "error",
                        "last_update": time.time(),
                        "error": str(e),
                        "pipeline_available": False,
                    }
                ),
                500,
            )

    @app.route("/routes")
    def list_routes():
        """Listar todas las rutas disponibles (solo desarrollo) MEJORADO"""
        if not app_config.debug:
            return jsonify({"error": "No disponible en producción"}), 403

        routes = []
        blueprint_routes = {}

        for rule in app.url_map.iter_rules():
            route_info = {
                "endpoint": rule.endpoint,
                "methods": sorted(list(rule.methods - {"HEAD", "OPTIONS"})),
                "rule": rule.rule,
            }

            # Agrupar por blueprint
            blueprint_name = rule.endpoint.split(".")[0] if "." in rule.endpoint else "main"
            if blueprint_name not in blueprint_routes:
                blueprint_routes[blueprint_name] = []
            blueprint_routes[blueprint_name].append(route_info)

            routes.append(route_info)

        return jsonify(
            {
                "routes": sorted(routes, key=lambda x: x["rule"]),
                "by_blueprint": blueprint_routes,
                "total": len(routes),
                "blueprints_registered": getattr(app, "blueprint_registration", []),
                "debug_mode": app_config.debug,
                "pipeline_available": _check_pipeline_available(),
            }
        )

    @app.route("/dashboard")
    def dashboard():
        """Ruta al dashboard (si existe template) MEJORADO"""
        try:
            # Obtener datos para el dashboard
            dashboard_data = _get_dashboard_data(app, app_config, logger)

            return render_template("dashboard.html", **dashboard_data)

        except Exception as e:
            logger.warning(f"Dashboard template no disponible: {e}")
            return jsonify(
                {
                    "message": "Dashboard no disponible - usando API mode",
                    "error": str(e),
                    "alternative": "/api/status",
                    "pipeline_available": _check_pipeline_available(),
                }
            )

    # NUEVA RUTA: Diagnóstico del sistema
    @app.route("/diagnose")
    def system_diagnose():
        """Diagnóstico completo del sistema"""
        try:
            diagnosis = {"timestamp": time.time(), "system_health": "unknown", "components": {}, "recommendations": []}

            # Diagnóstico de componentes
            try:
                llm_service = get_llm_service()
                llm_health = llm_service.health_check()
                diagnosis["components"]["llm_service"] = {"status": llm_health["status"], "details": llm_health["services"]}
            except Exception as e:
                diagnosis["components"]["llm_service"] = {"status": "error", "error": str(e)}
                diagnosis["recommendations"].append("Verificar configuración de LLM Service")

            # Diagnóstico del pipeline RAG
            try:
                rag_pipeline = get_rag_pipeline()
                if rag_pipeline:
                    pipeline_health = rag_pipeline.health_check()
                    diagnosis["components"]["rag_pipeline"] = {
                        "status": pipeline_health["status"],
                        "components": pipeline_health["components"],
                    }
                else:
                    diagnosis["components"]["rag_pipeline"] = {"status": "unavailable", "message": "Pipeline no inicializado"}
                    diagnosis["recommendations"].append("Verificar inicialización del Pipeline RAG")
            except Exception as e:
                diagnosis["components"]["rag_pipeline"] = {"status": "error", "error": str(e)}
                diagnosis["recommendations"].append("Revisar configuración del Pipeline RAG")

            # Diagnóstico de blueprints
            blueprint_status = getattr(app, "blueprint_registration", [])
            diagnosis["components"]["blueprints"] = {
                "registered": len([bp for bp in blueprint_status if bp["status"] == "success"]),
                "total": len(blueprint_status),
                "details": blueprint_status,
            }

            if diagnosis["components"]["blueprints"]["registered"] < diagnosis["components"]["blueprints"]["total"]:
                diagnosis["recommendations"].append("Algunos blueprints no se registraron correctamente")

            # Determinar salud general del sistema
            healthy_components = sum(1 for comp in diagnosis["components"].values() if comp.get("status") in ["healthy", "available"])
            total_components = len(diagnosis["components"])

            if healthy_components == total_components:
                diagnosis["system_health"] = "healthy"
            elif healthy_components > total_components / 2:
                diagnosis["system_health"] = "partial"
            else:
                diagnosis["system_health"] = "unhealthy"

            return jsonify(diagnosis)

        except Exception as e:
            logger.error(f"Error en diagnóstico: {e}")
            return jsonify({"error": "Error realizando diagnóstico", "details": str(e)}), 500

    logger.info("✅ Rutas configuradas correctamente")


def _check_pipeline_available():
    """Verificar si el pipeline RAG está disponible"""
    try:
        rag_pipeline = get_rag_pipeline()
        return rag_pipeline is not None
    except Exception:
        return False


def _get_dashboard_data(app, app_config, logger):
    """Obtener datos para el dashboard"""
    try:
        # Datos base
        dashboard_data = {
            "app_name": app_config.name,
            "app_version": getattr(app_config, "version", "1.2.0"),
            "usage_stats": {
                "queries_today": 0,
                "response_time_avg": 0,
                "success_rate": 0,
                "uptime_hours": round((time.time() - app.start_time) / 3600, 2) if hasattr(app, "start_time") else 0,
            },
            "performance_metrics": {
                "total_documents": 0,
                "vector_store_size": 0,
                "embedding_model": "unknown",
                "chunk_size": 500,
            },
            "providers_status": {},
            "pipeline_available": False,
        }

        # Enriquecer con datos del pipeline si está disponible
        if _check_pipeline_available():
            try:
                rag_pipeline = get_rag_pipeline()
                pipeline_stats = rag_pipeline.get_stats()

                dashboard_data.update(
                    {
                        "pipeline_available": True,
                        "usage_stats": {
                            "queries_today": pipeline_stats.get("queries_today", 0),
                            "response_time_avg": pipeline_stats.get("avg_response_time", 0),
                            "success_rate": pipeline_stats.get("success_rate", 0),
                            "uptime_hours": dashboard_data["usage_stats"]["uptime_hours"],
                        },
                        "performance_metrics": {
                            "total_documents": pipeline_stats.get("documents_count", 0),
                            "vector_store_size": pipeline_stats.get("vector_store_size_mb", 0),
                            "embedding_model": pipeline_stats.get("embedding_model", "unknown"),
                            "chunk_size": pipeline_stats.get("chunk_size", 500),
                        },
                    }
                )

                # Estado de proveedores desde el pipeline
                llm_providers = pipeline_stats.get("llm_providers", {})
                available_models = pipeline_stats.get("available_models", {})

                for provider, available in llm_providers.items():
                    dashboard_data["providers_status"][provider] = {
                        "available": available,
                        "status": "healthy" if available else "error",
                        "models": available_models.get(provider, []),
                        "model_count": len(available_models.get(provider, [])),
                    }

            except Exception as e:
                logger.error(f"Error obteniendo datos del pipeline para dashboard: {e}")

        # Fallback: obtener datos de servicios individuales
        if not dashboard_data["pipeline_available"]:
            try:
                llm_service = get_llm_service()
                llm_health = llm_service.health_check()

                # Actualizar con datos de LLM service
                for provider in ["ollama", "openai"]:
                    service_data = llm_health["services"].get(provider, {})
                    models_data = llm_health.get("models", {}).get(provider, [])

                    dashboard_data["providers_status"][provider] = {
                        "available": service_data.get("status") in ["available", "configured"],
                        "status": "healthy" if service_data.get("status") in ["available", "configured"] else "error",
                        "models": models_data,
                        "model_count": len(models_data),
                    }

            except Exception as e:
                logger.error(f"Error obteniendo datos de LLM service para dashboard: {e}")

        # Añadir datos de gráficos
        dashboard_data["charts_data"] = {
            "usage_over_time": {
                "labels": ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
                "datasets": [
                    {
                        "label": "Consultas",
                        "data": [12, 19, 15, 25, 22, 8, 14],
                        "borderColor": "rgb(75, 192, 192)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    }
                ],
            },
            "model_usage": {
                "labels": ["Ollama Local", "OpenAI", "Sin Respuesta"],
                "datasets": [{"data": [60, 35, 5], "backgroundColor": ["#ff6384", "#36a2eb", "#ffce56"]}],
            },
            "response_times": {
                "labels": ["< 1s", "1-2s", "2-5s", "> 5s"],
                "datasets": [{"data": [70, 20, 8, 2], "backgroundColor": ["#4bc0c0", "#ffcd56", "#ff9f40", "#ff6384"]}],
            },
        }

        return dashboard_data

    except Exception as e:
        logger.error(f"Error obteniendo datos para dashboard: {e}")
        return {"error": str(e)}


def setup_error_handlers(app, logger):
    """Configurar manejadores de errores MEJORADO"""

    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"Página no encontrada: {request.url}")

        if request.is_json or "application/json" in request.headers.get("Accept", ""):
            return (
                jsonify(
                    {
                        "error": "Página no encontrada",
                        "status": 404,
                        "available_routes": ["/dashboard", "/health", "/api/status", "/", "/chat"],
                        "pipeline_available": _check_pipeline_available(),
                    }
                ),
                404,
            )

        try:
            return render_template("errors/404.html", pipeline_available=_check_pipeline_available()), 404
        except Exception:
            return (
                f"""
        <div style="text-align: center; margin-top: 50px; font-family: Arial;">
            <h1>404 - Página no encontrada</h1>
            <p>La página que buscas no existe.</p>
            <div style="margin: 20px;">
                <a href="/" style="margin: 5px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">Inicio</a>
                <a href="/dashboard" style="margin: 5px; padding: 10px 15px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">Dashboard</a>
                <a href="/chat" style="margin: 5px; padding: 10px 15px; background: #17a2b8; color: white; text-decoration: none; border-radius: 5px;">Chat</a>
            </div>
            <small>Pipeline RAG: {'Disponible' if _check_pipeline_available() else 'No disponible'}</small>
        </div>
        """,
                404,
            )

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Error interno: {error}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        if request.is_json or "application/json" in request.headers.get("Accept", ""):
            return (
                jsonify(
                    {
                        "error": "Error interno del servidor",
                        "status": 500,
                        "debug": str(error) if app.config["DEBUG"] else "Error interno",
                        "pipeline_available": _check_pipeline_available(),
                    }
                ),
                500,
            )

        try:
            return render_template("errors/500.html", pipeline_available=_check_pipeline_available()), 500
        except Exception:
            return (
                f"""
        <div style="text-align: center; margin-top: 50px; font-family: Arial;">
            <h1>500 - Error interno del servidor</h1>
            <p>Se ha producido un error interno. Por favor, inténtalo más tarde.</p>
            <div style="margin: 20px;">
                <a href="/" style="margin: 5px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">Volver al inicio</a>
                <a href="/health" style="margin: 5px; padding: 10px 15px; background: #dc3545; color: white; text-decoration: none; border-radius: 5px;">Estado del Sistema</a>
            </div>
            {'<p><small>Debug: ' + str(error) + '</small></p>' if app.config.get('DEBUG') else ''}
            <small>Pipeline RAG: {'Disponible' if _check_pipeline_available() else 'No disponible'}</small>
        </div>
        """,
                500,
            )

    @app.errorhandler(413)
    def request_entity_too_large(error):
        logger.warning(f"Archivo demasiado grande subido desde {request.remote_addr}")

        if request.is_json or "application/json" in request.headers.get("Accept", ""):
            return jsonify({"error": "Archivo demasiado grande", "status": 413, "max_size": "16MB"}), 413

        return (
            """
    <div style="text-align: center; margin-top: 50px; font-family: Arial;">
        <h1>413 - Archivo demasiado grande</h1>
        <p>El archivo que intentas subir supera el límite de 16MB.</p>
        <a href="/" style="padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">Volver al inicio</a>
    </div>
    """,
            413,
        )


def setup_context_processors(app, app_config):
    """Configurar procesadores de contexto MEJORADO"""

    @app.context_processor
    def inject_global_vars():
        return {
            "app_name": app_config.name,
            "app_version": getattr(app_config, "version", "1.2.0"),
            "app_description": app_config.description,
            "current_year": "2025",
            "debug_mode": app_config.debug,
            "pipeline_available": _check_pipeline_available(),
            "environment": getattr(app_config, "environment", "development"),
        }


def verify_system_status(logger):
    """Verificar estado inicial del sistema MEJORADO"""
    logger.info("🔍 Verificando estado inicial del sistema...")
    system_status = {
        "llm_service": False,
        "rag_pipeline": False,
        "components": {},
        "warnings": [],
        "errors": [],
    }

    # Verificar LLM Service
    try:
        llm_service = get_llm_service()
        health = llm_service.health_check()
        system_status["llm_service"] = health["status"] == "healthy"
        system_status["components"]["llm"] = health["services"]

        if system_status["llm_service"]:
            logger.info(f"✅ LLM Service disponible - Estado: {health['status']}")

            # Detallar proveedores disponibles
            for provider, info in health["services"].items():
                status = info.get("status", "unknown")
                if status in ["available", "configured"]:
                    models = health.get("models", {}).get(provider, [])
                    logger.info(f"   ✅ {provider.title()}: {len(models)} modelos disponibles")
                else:
                    logger.warning(f"   ⚠️ {provider.title()}: {status}")
                    system_status["warnings"].append(f"{provider.title()} no disponible: {status}")
        else:
            logger.error(f"❌ LLM Service con problemas - Estado: {health['status']}")
            system_status["errors"].append("LLM Service no está completamente operativo")

    except Exception as e:
        logger.error(f"❌ Error verificando LLM Service: {e}")
        system_status["errors"].append(f"LLM Service: {str(e)}")

    # Verificar Pipeline RAG
    try:
        rag_pipeline = get_rag_pipeline()
        if rag_pipeline:
            pipeline_health = rag_pipeline.health_check()
            pipeline_stats = rag_pipeline.get_stats()

            system_status["rag_pipeline"] = pipeline_health["status"] in ["healthy", "available"]
            system_status["components"]["pipeline"] = pipeline_health["components"]

            if system_status["rag_pipeline"]:
                docs_count = pipeline_stats.get("documents_count", 0)
                vector_store = pipeline_stats.get("vector_store_type", "Unknown")
                embedding_model = pipeline_stats.get("embedding_model", "Unknown")

                logger.info("✅ Pipeline RAG disponible y listo")
                logger.info(f"   📊 {docs_count} documentos indexados")
                logger.info(f"   🗄️ Vector Store: {vector_store}")
                logger.info(f"   🧠 Embedding Model: {embedding_model}")

                if docs_count == 0:
                    system_status["warnings"].append("Pipeline RAG sin documentos indexados")
            else:
                logger.warning(f"⚠️ Pipeline RAG con advertencias - Estado: {pipeline_health['status']}")
                system_status["warnings"].append(f"Pipeline RAG: {pipeline_health['status']}")
        else:
            logger.warning("⚠️ Pipeline RAG no inicializado")
            system_status["warnings"].append("Pipeline RAG no está inicializado")

    except ImportError:
        logger.info("📝 Pipeline RAG no disponible (módulo no encontrado)")
        system_status["warnings"].append("Pipeline RAG no instalado/configurado")
    except Exception as e:
        logger.error(f"❌ Error verificando Pipeline RAG: {e}")
        system_status["errors"].append(f"Pipeline RAG: {str(e)}")

    # Resumen del estado del sistema
    total_components = len([k for k in ["llm_service", "rag_pipeline"] if k])
    healthy_components = sum([system_status["llm_service"], system_status["rag_pipeline"]])

    if healthy_components == total_components:
        overall_status = "✅ SISTEMA COMPLETAMENTE OPERATIVO"
        logger.info(overall_status)
    elif healthy_components > 0:
        overall_status = "⚠️ SISTEMA PARCIALMENTE OPERATIVO"
        logger.warning(overall_status)
        logger.warning(f"   Componentes activos: {healthy_components}/{total_components}")
    else:
        overall_status = "❌ SISTEMA CON PROBLEMAS CRÍTICOS"
        logger.error(overall_status)

    # Mostrar advertencias y errores
    if system_status["warnings"]:
        logger.warning("📋 Advertencias del sistema:")
        for warning in system_status["warnings"]:
            logger.warning(f"   ⚠️ {warning}")

    if system_status["errors"]:
        logger.error("❌ Errores del sistema:")
        for error in system_status["errors"]:
            logger.error(f"   ❌ {error}")

    return system_status


def print_startup_info(app_config, registered_blueprints, system_status):
    """Mostrar información de inicio MEJORADA"""
    print("\n" + "=" * 80)
    print("🚀 PROTOTIPO_CHATBOT TFM - APLICACIÓN LISTA")
    print("=" * 80)

    # URLs disponibles
    print("🌐 URLs del sistema:")
    base_url = f"http://{app_config.host}:{app_config.port}"

    main_urls = [
        ("", "Página principal"),
        ("/dashboard", "Dashboard completo con métricas"),
        ("/chat", "Interfaz de chat RAG"),
        ("/health", "Estado del sistema (JSON)"),
        ("/api/status", "API de estado detallado"),
        ("/diagnose", "Diagnóstico completo del sistema"),
    ]

    for path, description in main_urls:
        print(f"   {base_url}{path:<20} - {description}")

    if app_config.debug:
        print(f"   {base_url}/routes             - Lista de rutas (DEBUG)")

    # Estado del sistema
    print(f"\n🔧 Estado del sistema:")
    print(f"   🎯 Blueprints registrados: {registered_blueprints}")
    print(f"   🎨 Templates: app/templates/")
    print(f"   📊 Endpoints API: /health, /api/status, /ajax/quick-stats")
    print(f"   🔍 Modo debug: {'Activado' if app_config.debug else 'Desactivado'}")
    print(f"   🏗️ Pipeline RAG: {'Disponible' if system_status.get('rag_pipeline') else 'No disponible'}")
    print(f"   🤖 LLM Service: {'Operativo' if system_status.get('llm_service') else 'Limitado'}")

    # Funcionalidades disponibles
    print(f"\n💡 Funcionalidades:")
    features = [
        "🤖 Sistema LLM dual (Ollama + OpenAI)",
        "📊 Dashboard con métricas en tiempo real",
        "🔄 Health checks automáticos y diagnóstico",
        "📈 Estadísticas detalladas y exportación",
        "🎨 Interface web responsive y moderna",
    ]

    if system_status.get("rag_pipeline"):
        features.extend(
            [
                "🔍 Pipeline RAG completo integrado",
                "📚 Ingesta multimodal de documentos",
                "⚖️ Comparación de modelos automatizada",
                "🗃️ Vector stores duales (FAISS/ChromaDB)",
            ]
        )

    for feature in features:
        print(f"   {feature}")

    # Advertencias si las hay
    if system_status.get("warnings"):
        print(f"\n⚠️ Advertencias:")
        for warning in system_status["warnings"]:
            print(f"   ⚠️ {warning}")

    # Errores si los hay
    if system_status.get("errors"):
        print(f"\n❌ Errores:")
        for error in system_status["errors"]:
            print(f"   ❌ {error}")

    # Información adicional
    print(f"\n📋 Información técnica:")
    print(f"   📁 Directorio: {project_root}")
    print(f"   🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"   🔧 Flask: Configurado con {registered_blueprints} blueprints")
    print(f"   📝 Versión: {getattr(app_config, 'version', '1.2.0')}")
    print(f"   🏷️ Entorno: {getattr(app_config, 'environment', 'development')}")

    print(f"\n⚠️  Usa Ctrl+C para detener el servidor")
    print("=" * 80)


def main():
    """Función principal de arranque MEJORADA"""
    print("🚀 Iniciando Prototipo_chatbot TFM...")
    print("📁 Directorio del proyecto:", project_root)
    print("👨‍🎓 Vicente Caruncho Ramos - Sistemas Inteligentes UJI")
    print("🔗 GitHub: https://github.com/cholinyo/Prototipo_chatbot")
    print("-" * 80)

    try:
        # Crear archivos faltantes
        create_missing_files()

        # Verificar dependencias críticas
        try:
            import flask  # noqa: F401
            print("✅ Flask importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando Flask: {e}")
            print("💡 Instala las dependencias: pip install -r requirements.txt")
            sys.exit(1)

        # Crear aplicación Flask
        app, app_config, logger = create_flask_app()

        # Marcar tiempo de inicio
        app.start_time = time.time()

        # Registrar blueprints
        registered_blueprints = register_blueprints(app)

        # Configurar rutas principales
        setup_routes(app, app_config, logger)

        # Configurar manejadores de errores
        setup_error_handlers(app, logger)

        # Configurar procesadores de contexto
        setup_context_processors(app, app_config)

        logger.info("Aplicación Flask configurada exitosamente")

        # Verificar estado inicial del sistema
        system_status = verify_system_status(logger)

        # Mostrar información de inicio
        print_startup_info(app_config, registered_blueprints, system_status)

        # Iniciar servidor
        logger.info(f"🌐 Servidor iniciando en http://{app_config.host}:{app_config.port}")

        # Verificar si hay problemas críticos antes de iniciar
        if system_status.get("errors") and not system_status.get("llm_service"):
            logger.warning("⚠️ Iniciando con errores críticos - funcionalidad limitada")

        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            use_reloader=True,
            threaded=True,
        )

    except KeyboardInterrupt:
        print("\n⏹️  Aplicación detenida por el usuario")
        print("👋 ¡Hasta luego! Gracias por usar Prototipo_chatbot TFM")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error crítico iniciando la aplicación:")
        print(f"🔍 Error: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        print("\n💡 Soluciones posibles:")
        print("   1. pip install -r requirements.txt")
        print("   2. Verificar estructura de directorios")
        print("   3. Revisar archivos de configuración en config/")
        print("   4. Verificar permisos de escritura en data/ y logs/")
        print("   5. Consultar documentación en /docs")
        print("   6. Revisar logs en logs/ para más detalles")
        sys.exit(1)

# Actualizar la función register_blueprints en run.py

def register_blueprints(app):
    """Registrar blueprints de la aplicación"""
    try:
        # Blueprint principal
        from app.routes.main import main_bp
        app.register_blueprint(main_bp)
        print("✅ Blueprint main registrado")
        
        # Blueprint API para fuentes de datos
        try:
            from app.routes.api.data_sources import data_sources_api
            app.register_blueprint(data_sources_api)
            print("✅ Blueprint data_sources_api registrado")
        except ImportError as e:
            print(f"⚠️ Blueprint data_sources_api no disponible: {e}")
        
        # Blueprint API para chat (si existe)
        try:
            from app.routes.api.chat import chat_api
            app.register_blueprint(chat_api)
            print("✅ Blueprint chat_api registrado")
        except ImportError as e:
            print(f"⚠️ Blueprint chat_api no disponible: {e}")
        
        # Blueprint API para comparación (si existe)
        try:
            from app.routes.api.comparison import comparison_api
            app.register_blueprint(comparison_api)
            print("✅ Blueprint comparison_api registrado")
        except ImportError as e:
            print(f"⚠️ Blueprint comparison_api no disponible: {e}")
            
    except Exception as e:
        print(f"❌ Error registrando blueprints: {e}")
        # Registrar solo main como fallback
        try:
            from app.routes.main import main_bp
            app.register_blueprint(main_bp)
            print("✅ Blueprint main registrado como fallback")
        except Exception as fallback_error:
            print(f"❌ Error crítico registrando blueprint main: {fallback_error}")

# También agregar la creación de directorios necesarios en create_missing_files():

def create_missing_files():
    """Crear archivos y directorios faltantes"""
    try:
        # Directorios necesarios
        directories = [
            'app', 'app/routes', 'app/routes/api', 'app/core', 'app/services', 
            'app/models', 'app/templates', 'app/static', 'app/static/css', 
            'app/static/js', 'config', 'data', 'data/vectorstore', 
            'data/vectorstore/faiss', 'data/vectorstore/chromadb',
            'data/ingestion',  # <- Nuevo directorio para almacenar datos de ingesta
            'data/cache', 'data/cache/embeddings', 'logs', 'tests'
        ]
        
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Archivos __init__.py necesarios
        init_files = [
            'app/__init__.py',
            'app/routes/__init__.py', 
            'app/routes/api/__init__.py',
            'app/core/__init__.py',
            'app/services/__init__.py',
            'app/models/__init__.py'
        ]
        
        for init_file in init_files:
            init_path = project_root / init_file
            if not init_path.exists():
                init_path.write_text('"""Package initialization"""')
        
        print("✅ Directorios y archivos necesarios creados")
        
    except Exception as e:
        print(f"⚠️ Error creando archivos: {e}")

if __name__ == "__main__":
    main()

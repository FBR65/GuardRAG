#!/usr/bin/env python3
"""
Shared COLPALI Manager
Verwaltet eine einzelne COLPALI-Instanz für alle RAG-Agenten
"""

import logging
from typing import Optional, Dict, Any
from .colpali_integration import COLPALIProcessor
from .qdrant_integration import QdrantConfig

logger = logging.getLogger(__name__)


class COLPALIManager:
    """
    Singleton Manager für COLPALI-Instanzen.
    Verhindert das mehrfache Laden des gleichen Modells.
    """

    _instances: Dict[str, COLPALIProcessor] = {}
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(
        cls,
        model_name: str,
        device: str,
        qdrant_config: QdrantConfig,
        force_reload: bool = False,
    ) -> COLPALIProcessor:
        """
        Holt oder erstellt eine COLPALI-Instanz.

        Args:
            model_name: Name des COLPALI-Modells
            device: Device (cuda/cpu)
            qdrant_config: Qdrant-Konfiguration
            force_reload: Forciert das Neuladen des Modells

        Returns:
            COLPALIProcessor-Instanz
        """
        # Erstelle einen eindeutigen Schlüssel für diese Konfiguration
        config_key = f"{model_name}_{device}_{qdrant_config.host}_{qdrant_config.port}"

        # Prüfe ob bereits eine Instanz mit dieser Konfiguration existiert
        if config_key in cls._instances and not force_reload:
            logger.info(f"♻️ Wiederverwendung existierender COLPALI Instanz: {config_key}")
            return cls._instances[config_key]

        # Erstelle neue Instanz
        logger.info(f"� Erstelle neue COLPALI Instanz: {config_key}")

        try:
            colpali_instance = COLPALIProcessor(
                model_name=model_name,
                device=device,
                qdrant_config=qdrant_config,
            )

            # Speichere Instanz und Konfiguration
            cls._instances[config_key] = colpali_instance
            cls._configs[config_key] = {
                "model_name": model_name,
                "device": device,
                "qdrant_config": qdrant_config,
                "created_at": __import__("datetime").datetime.now(),
            }

            logger.info(f"✅ COLPALI Instanz erfolgreich erstellt: {config_key}")
            return colpali_instance

        except Exception as e:
            logger.error(f"❌ Fehler beim Erstellen der COLPALI Instanz: {e}")
            raise

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Gibt Statistiken über die verwalteten Instanzen zurück."""
        return {
            "total_instances": len(cls._instances),
            "instance_keys": list(cls._instances.keys()),
            "configs": cls._configs,
            "memory_usage": cls._get_memory_usage(),
        }

    @classmethod
    def _get_memory_usage(cls) -> Dict[str, str]:
        """Geschätzte Speichernutzung der Instanzen."""
        memory_info = {}
        for key, instance in cls._instances.items():
            try:
                # Grobe Schätzung basierend auf Modellgröße
                if hasattr(instance, "model") and hasattr(
                    instance.model, "num_parameters"
                ):
                    params = instance.model.num_parameters()
                    # Annahme: 4 Bytes pro Parameter (float32)
                    memory_mb = (params * 4) / (1024 * 1024)
                    memory_info[key] = f"~{memory_mb:.0f} MB"
                else:
                    memory_info[key] = "Unknown"
            except Exception:
                memory_info[key] = "Error"
        return memory_info

    @classmethod
    def clear_cache(cls, model_name: Optional[str] = None):
        """
        Löscht den Cache.

        Args:
            model_name: Spezifisches Modell löschen, oder None für alle
        """
        if model_name:
            # Lösche spezifisches Modell
            keys_to_remove = [k for k in cls._instances.keys() if model_name in k]
            for key in keys_to_remove:
                del cls._instances[key]
                del cls._configs[key]
                logger.info(f"🗑️ Removed COLPALI instance: {key}")
        else:
            # Lösche alle
            cls._instances.clear()
            cls._configs.clear()
            logger.info("🗑️ Cleared all COLPALI instances")

    @classmethod
    def reload_instance(
        cls, model_name: str, device: str, qdrant_config: QdrantConfig
    ) -> COLPALIProcessor:
        """Forciert das Neuladen einer spezifischen Instanz."""
        return cls.get_instance(model_name, device, qdrant_config, force_reload=True)

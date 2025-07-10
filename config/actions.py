# Custom Actions für NeMo Guardrails
# Implementiert spezifische Sicherheitslogik für das RAG-System

from typing import Optional
import re
import logging
from nemoguardrails.actions import action

logger = logging.getLogger(__name__)


@action(is_system_action=True)
async def check_harmful_content(context: Optional[dict] = None) -> bool:
    """
    Überprüft ob Eingabe schädliche Inhalte enthält.
    Nutzt KI-basierte Erkennung statt naive Keyword-Listen.
    """
    user_message = context.get("user_message", "")

    # Patterns für eindeutig schädliche Inhalte (sehr spezifisch)
    harmful_patterns = [
        r"\b(suicide|kill\s+yourself|harm\s+yourself)\b",
        r"\b(bomb|terrorist|attack)\s+(plan|instruction|how\s+to)\b",
        r"\b(child\s+)(abuse|exploitation|pornography)\b",
        r"\b(steal|hack|illegal)\s+(credit\s+card|password|account)\b",
    ]

    for pattern in harmful_patterns:
        if re.search(pattern, user_message.lower()):
            logger.warning(f"Harmful content detected: {pattern}")
            return True

    return False


@action(is_system_action=True)
async def check_data_extraction(context: Optional[dict] = None) -> bool:
    """
    Erkennt Versuche, vollständige Datenbestände zu extrahieren.
    """
    user_message = context.get("user_message", "")

    extraction_patterns = [
        r"\b(give\s+me\s+all|show\s+me\s+all|list\s+all|export\s+all)\b",
        r"\b(download\s+all|dump\s+all|extract\s+all)\b",
        r"\b(all\s+(documents|files|data|content))\b",
        r"\b(complete\s+(database|dataset|index))\b",
    ]

    for pattern in extraction_patterns:
        if re.search(pattern, user_message.lower()):
            logger.warning(f"Data extraction attempt detected: {pattern}")
            return True

    return False


@action(is_system_action=True)
async def check_prompt_injection(context: Optional[dict] = None) -> bool:
    """
    Erkennt Prompt-Injection-Versuche.
    """
    user_message = context.get("user_message", "")

    injection_patterns = [
        r"\b(ignore\s+(previous|above|system))\b",
        r"\b(system\s*(prompt|instruction|role))\b",
        r"\b(act\s+as\s+if|pretend\s+to\s+be)\b",
        r"\b(disregard\s+(safety|rules|guidelines))\b",
        r"(\[SYSTEM\]|\[INST\]|\<\|system\|\>)",
    ]

    for pattern in injection_patterns:
        if re.search(pattern, user_message.lower()):
            logger.warning(f"Prompt injection detected: {pattern}")
            return True

    return False


@action(is_system_action=True)
async def check_topic_relevance(context: Optional[dict] = None) -> bool:
    """
    Überprüft ob Anfrage zum RAG-System-Zweck passt.
    """
    user_message = context.get("user_message", "")

    # Themen die eindeutig off-topic sind
    off_topic_patterns = [
        r"\b(weather|stock\s+market|sports\s+scores)\b",
        r"\b(cooking\s+recipe|restaurant\s+recommendation)\b",
        r"\b(dating\s+advice|relationship\s+help)\b",
        r"\b(medical\s+diagnosis|legal\s+advice)\b",
    ]

    for pattern in off_topic_patterns:
        if re.search(pattern, user_message.lower()):
            logger.info(f"Off-topic request detected: {pattern}")
            return True

    return False


@action(is_system_action=True)
async def check_personal_data(context: Optional[dict] = None) -> bool:
    """
    Erkennt persönliche Daten in der Eingabe.
    """
    user_message = context.get("user_message", "")

    # Patterns für persönliche Daten
    personal_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}\s+\d{4}\s+\d{4}\s+\d{4}\b",  # Credit Card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\+?[\d\s\-\(\)]{10,}\b",  # Phone numbers
    ]

    for pattern in personal_patterns:
        if re.search(pattern, user_message):
            logger.warning("Personal data detected in input")
            return True

    return False


@action(is_system_action=True)
async def check_harmful_output(context: Optional[dict] = None) -> bool:
    """
    Überprüft ob Ausgabe schädliche Inhalte enthält.
    """
    bot_response = context.get("bot_response", "")

    # Ähnliche Patterns wie bei Input, aber für Output
    harmful_patterns = [
        r"\b(suicide|self-harm|violence)\s+(instruction|method|way)\b",
        r"\b(illegal\s+drug|weapon)\s+(making|manufacturing)\b",
    ]

    for pattern in harmful_patterns:
        if re.search(pattern, bot_response.lower()):
            logger.warning(f"Harmful output detected: {pattern}")
            return True

    return False


@action(is_system_action=True)
async def check_data_leakage(context: Optional[dict] = None) -> bool:
    """
    Überprüft ob Ausgabe sensitive Daten preisgibt.
    """
    bot_response = context.get("bot_response", "")

    # Patterns für potentielle Datenlecks
    leakage_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\bpassword\s*:\s*\w+\b",  # Passwords
        r"\bapi[_\s]key\s*:\s*\w+\b",  # API Keys
    ]

    for pattern in leakage_patterns:
        if re.search(pattern, bot_response):
            logger.warning("Potential data leakage in output")
            return True

    return False

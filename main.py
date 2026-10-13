import hmac
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
from supabase import Client, create_client

from agent import detect_payment_intent, generate_reply


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("isley_agent")
app = FastAPI(title="ISLEY POS Sales Agent", version="1.0.0")
PROJECT_ID = "isley_pos"


@dataclass
class InboundMessage:
    platform: str
    social_id: str
    phone: str | None
    text: str
    has_media: bool
    media_url: str | None
    display_name: str | None
    message_id: str | None


def _supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY son obligatorios")
    return create_client(url, key)


def _walk_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("conversation", "text", "caption", "body", "title"):
            if isinstance(value.get(key), str):
                return value[key]
        for key, child in value.items():
            if key.lower() in {"url", "link", "directpath", "mediaurl", "mimetype", "file"}:
                continue
            if not isinstance(child, (dict, list)):
                continue
            result = _walk_text(child)
            if result:
                return result
    if isinstance(value, list):
        for child in value:
            result = _walk_text(child)
            if result:
                return result
    return ""


def _media_details(message: Any) -> tuple[bool, str | None]:
    if not isinstance(message, dict):
        return False, None
    for key in ("imageMessage", "documentMessage", "videoMessage", "audioMessage", "image", "attachments"):
        media = message.get(key)
        if media:
            if isinstance(media, list):
                media = media[0] if media else {}
            url = media.get("url") or media.get("link") if isinstance(media, dict) else None
            return True, url
    return False, None


def _phone_from_social_id(social_id: str, platform: str) -> str | None:
    if platform != "whatsapp":
        return None
    number = social_id.split("@", 1)[0].split(":", 1)[0]
    return number if number.isdigit() else None


def _normalize_evolution(payload: dict[str, Any]) -> list[InboundMessage]:
    event = str(payload.get("event", "")).lower()
    if event and not event.endswith("messages.upsert"):
        return []
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        return []
    key = data.get("key") or {}
    if not isinstance(key, dict) or key.get("fromMe"):
        return []
    social_id = str(key.get("remoteJid") or "")
    if not social_id or social_id.endswith("@g.us"):
        return []
    message = data.get("message") or {}
    has_media, media_url = _media_details(message)
    return [InboundMessage(
        platform="whatsapp",
        social_id=social_id,
        phone=_phone_from_social_id(social_id, "whatsapp"),
        text=_walk_text(message),
        has_media=has_media,
        media_url=media_url,
        display_name=data.get("pushName"),
        message_id=key.get("id"),
    )]


def _normalize_meta(payload: dict[str, Any]) -> list[InboundMessage]:
    results: list[InboundMessage] = []
    for entry in payload.get("entry", []) if isinstance(payload.get("entry"), list) else []:
        if not isinstance(entry, dict):
            continue
        for event in entry.get("messaging", []) if isinstance(entry.get("messaging"), list) else []:
            if not isinstance(event, dict):
                continue
            message = event.get("message") or {}
            if not isinstance(message, dict) or message.get("is_echo"):
                continue
            sender = event.get("sender") or {}
            social_id = str(sender.get("id") or "")
            if not social_id:
                continue
            attachments = message.get("attachments") or []
            first_attachment = attachments[0] if attachments else {}
            results.append(InboundMessage(
                platform=(
                    "instagram"
                    if payload.get("object") == "instagram"
                    else "messenger"
                    if payload.get("object") == "page"
                    else "facebook"
                ),
                social_id=social_id,
                phone=None,
                text=str(message.get("text") or ""),
                has_media=bool(attachments),
                media_url=(first_attachment.get("payload") or {}).get("url") if isinstance(first_attachment, dict) else None,
                display_name=None,
                message_id=(message.get("mid") or event.get("postback", {}).get("mid")),
            ))
    return results


def normalize_webhook(payload: dict[str, Any]) -> list[InboundMessage]:
    if "entry" in payload:
        return _normalize_meta(payload)
    return _normalize_evolution(payload)


def _persist_handoff(message: InboundMessage) -> None:
    table = os.getenv("SUPABASE_CONVERSATIONS_TABLE", "conversations")
    record = {
        "phone": message.phone,
        "social_id": message.social_id,
        "project_id": PROJECT_ID,
        "status": "pendiente_verificacion_pago",
    }
    client = _supabase()
    query = client.table(table).update(record).eq("project_id", PROJECT_ID).eq("social_id", message.social_id)
    existing = query.execute()
    if not existing.data:
        client.table(table).insert(record).execute()


def _is_bot_paused(message: InboundMessage) -> bool:
    table = os.getenv("SUPABASE_CONVERSATIONS_TABLE", "conversations")
    result = (
        _supabase()
        .table(table)
        .select("status")
        .eq("project_id", PROJECT_ID)
        .eq("social_id", message.social_id)
        .limit(1)
        .execute()
    )
    return bool(result.data and result.data[0].get("status") == "pendiente_verificacion_pago")


async def _notify_admin(message: InboundMessage, reason: str) -> None:
    alert = {
        "event": "payment_handoff",
        "project_id": PROJECT_ID,
        "status": "pendiente_verificacion_pago",
        "reason": reason,
        "platform": message.platform,
        "phone": message.phone,
        "social_id": message.social_id,
        "display_name": message.display_name,
        "message": message.text,
        "capture_url": message.media_url,
        "message_id": message.message_id,
    }
    logger.warning("ADMIN_PAYMENT_HANDOFF %s", json.dumps(alert, ensure_ascii=False))
    url = os.getenv("ADMIN_ALERT_WEBHOOK_URL")
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=alert)
            response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("No se pudo entregar la alerta al webhook administrativo")


async def _send_reply(message: InboundMessage, text: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if message.platform == "whatsapp":
                base_url = os.getenv("EVOLUTION_API_URL", "").rstrip("/")
                instance = os.getenv("EVOLUTION_INSTANCE", "")
                api_key = os.getenv("EVOLUTION_API_KEY", "")
                if not (base_url and instance and api_key and message.phone):
                    logger.info("Respuesta generada, pero Evolution API no está configurada")
                    return
                response = await client.post(
                    f"{base_url}/message/sendText/{instance}",
                    headers={"apikey": api_key},
                    json={"number": message.phone, "text": text},
                )
            else:
                graph_version = os.getenv("META_GRAPH_VERSION", "v21.0")
                if message.platform == "instagram":
                    account_id = os.getenv("META_INSTAGRAM_ACCOUNT_ID", "")
                    access_token = os.getenv("META_INSTAGRAM_ACCESS_TOKEN", os.getenv("META_PAGE_ACCESS_TOKEN", ""))
                else:
                    account_id = os.getenv("META_PAGE_ID", "")
                    access_token = os.getenv("META_PAGE_ACCESS_TOKEN", "")
                if not (account_id and access_token):
                    logger.info("Respuesta generada, pero Meta no está configurado para %s", message.platform)
                    return
                response = await client.post(
                    f"https://graph.facebook.com/{graph_version}/{account_id}/messages",
                    params={"access_token": access_token},
                    json={"recipient": {"id": message.social_id}, "message": {"text": text}},
                )
            response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("No se pudo enviar la respuesta al canal %s", message.platform)


def _valid_webhook_secret(provided: str | None) -> bool:
    expected = os.getenv("WEBHOOK_SECRET")
    if not expected:
        return False
    return bool(provided and hmac.compare_digest(provided, expected))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "project_id": PROJECT_ID}


@app.get("/webhook")
async def verify_meta_webhook(request: Request) -> PlainTextResponse:
    params = request.query_params
    verify_token = os.getenv("META_VERIFY_TOKEN", "")
    if params.get("hub.mode") == "subscribe" and verify_token and hmac.compare_digest(params.get("hub.verify_token", ""), verify_token):
        return PlainTextResponse(params.get("hub.challenge", ""))
    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict[str, Any]:
    body = await request.body()
    meta_secret = os.getenv("META_APP_SECRET")
    meta_signature_valid = False
    if meta_secret and x_hub_signature_256:
        expected = "sha256=" + hmac.new(meta_secret.encode(), body, hashlib.sha256).hexdigest()
        meta_signature_valid = hmac.compare_digest(x_hub_signature_256, expected)
    if not meta_signature_valid and not _valid_webhook_secret(x_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid or unconfigured webhook signature")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Webhook JSON must be an object")

    messages = normalize_webhook(payload)
    for message in messages:
        try:
            paused = await run_in_threadpool(_is_bot_paused, message)
        except Exception:
            logger.exception("No se pudo consultar el estado de la conversación %s", message.social_id)
            raise HTTPException(status_code=503, detail="Could not read conversation status") from None
        if paused:
            logger.info("IA pausada para conversación %s", message.social_id)
            continue

        handoff, reason = detect_payment_intent(message.text, message.has_media)
        if handoff:
            try:
                await run_in_threadpool(_persist_handoff, message)
            except Exception:
                logger.exception("Falló la pausa/persistencia de conversación %s", message.social_id)
                raise HTTPException(status_code=503, detail="Could not persist payment handoff") from None
            await _notify_admin(message, reason or "pago_reportado")
            await _send_reply(message, "Gracias por avisarme. Tu conversación quedó en revisión; el administrador verificará el pago y te confirmará los próximos pasos. No compartas claves ni códigos de seguridad por este chat.")
            continue

        try:
            reply = await generate_reply(message.text, message.has_media)
        except Exception:
            logger.exception("Falló la generación de respuesta para %s", message.social_id)
            reply = "Gracias por escribir a ISLEY POS. En este momento no puedo responder con precisión; cuéntame qué necesitas saber y te atenderemos pronto."
        await _send_reply(message, reply)

    return {"ok": True, "processed": len(messages)}
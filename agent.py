import json
import os
import unicodedata
from typing import Any

from openai import AsyncOpenAI


PRODUCT_CONTEXT = """ISLEY POS es un punto de venta SaaS web e híbrido para negocios de Venezuela.
Funciona online y offline. Incluye inventario ordenado alfabéticamente con alertas de
reabastecimiento en rojo, importación masiva desde Excel, cálculo automático de costos
por lote e IVA, restaurante y bar con comandas y mesas, cuentas de crédito, notas de
despacho, presupuestos que no descuentan inventario y facturación con cliente genérico.
La suscripción estándar cuesta USD 9.99 al mes y existe una promoción para pequeños
negocios de USD 5 al mes. No hay demos públicas antes de comprar; después de la compra
se configura un entorno demo limpio y aislado. Las tasas de cambio las administra
estrictamente el rol Administrador.
Métodos: en bolívares, Pago Móvil, transferencia bancaria y Punto de Venta; en dólares,
Zelle y Binance Pay; efectivo en USD o Bs. Nunca inventes datos bancarios, tasas,
credenciales, funcionalidades, descuentos ni condiciones."""

SYSTEM_PROMPT = """Eres el agente comercial de ISLEY POS para Venezuela. Atiende con amabilidad,
brevedad y vocabulario comercial local natural. Responde solo con la información
confirmada en el contexto. No afirmes que verificas pagos, no concedas accesos, no
solicites claves ni datos sensibles. Si aún no escogió método de pago, presenta los
métodos disponibles y pregunta cuál prefiere. No prometas demo antes de la compra.
Si piden detalles que no están disponibles, dilo claramente y ofrece que el administrador
los confirme. El sistema detiene automáticamente la IA y deriva al administrador cuando
detecta intención de pago o comprobante; no intentes evitar esa derivación.

CONTEXTO:
{product_context}

INSTRUCCIÓN DE PAGO CONFIGURADA (si existe; usa únicamente el dato correspondiente):
{payment_instruction}"""


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def detect_payment_intent(text: str, has_media: bool = False) -> tuple[bool, str | None]:
    folded = _fold(text)
    proof_terms = ("comprobante", "capture", "captura", "voucher", "recibo de pago", "pago realizado")
    paid_terms = ("ya pague", "pague ya", "acabo de pagar", "realice el pago", "hice el pago", "ya hice el pago", "pago listo")
    about_to_pay_terms = ("voy a pagar", "procedere a pagar", "estoy por pagar", "voy pagando", "pago ahora", "listo para pagar", "lista para pagar", "quiero pagar ya")

    if any(term in folded for term in proof_terms) or (has_media and not folded.strip()):
        return True, "comprobante_o_captura"
    if any(term in folded for term in paid_terms):
        return True, "pago_reportado"
    if any(term in folded for term in about_to_pay_terms):
        return True, "intencion_inminente_de_pago"
    return False, None


def _payment_instruction(text: str) -> str:
    folded = _fold(text)
    methods = {
        "pago_movil": ("pago movil",),
        "transferencia": ("transferencia", "transferencia bancaria"),
        "punto_venta": ("punto de venta", "punto venta", "punto de pago"),
        "zelle": ("zelle",),
        "binance_pay": ("binance pay", "binance"),
        "efectivo_usd": ("efectivo en dolares", "efectivo usd", "efectivo en usd"),
        "efectivo_bs": ("efectivo en bolivares", "efectivo bs", "efectivo en bs"),
    }
    selected = next((key for key, phrases in methods.items() if any(phrase in folded for phrase in phrases)), None)
    if selected is None:
        return "No indicó un método específico. Informa solo los métodos disponibles y pregúntale cuál prefiere."

    try:
        configured = json.loads(os.getenv("PAYMENT_INSTRUCTIONS_JSON", "{}"))
    except json.JSONDecodeError:
        configured = {}
    instruction = configured.get(selected) if isinstance(configured, dict) else None
    if isinstance(instruction, str) and instruction.strip():
        return f"Método {selected}: {instruction.strip()}"
    return f"Eligió {selected}, pero no hay instrucciones de cobro configuradas para ese método. Indícale que el administrador debe confirmar los datos; no los inventes."


async def generate_reply(text: str, has_media: bool = False) -> str:
    handoff, _ = detect_payment_intent(text, has_media)
    if handoff:
        return "Gracias por avisarme. Voy a pasar tu caso al administrador para que revise el pago y te confirme los próximos pasos. No compartas claves ni códigos de seguridad por este chat."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Gracias por escribir a ISLEY POS. ¿Qué te gustaría saber del sistema o cuál es el tipo de negocio que tienes?"

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        max_tokens=350,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(
                product_context=PRODUCT_CONTEXT,
                payment_instruction=_payment_instruction(text),
            )},
            {"role": "user", "content": text or "El cliente envió un archivo sin texto. Pregunta qué necesita consultar."},
        ],
    )
    answer = response.choices[0].message.content
    return answer.strip() if answer else "¿Me cuentas un poco más para poder orientarte?"
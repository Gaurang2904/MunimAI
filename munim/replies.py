"""Final replies to the merchant, in their language.

The LLM picks the outcome; the wording comes from reviewed templates, so a small model can never send a
garbled or over-promising message about money.
"""
from datetime import date

T = {
    "payment_received": {
        "hi": "₹{amount} आपके खाते में आ गए हैं।",
        "mr": "₹{amount} तुमच्या खात्यात जमा झाले आहेत.",
        "en": "₹{amount} has reached your account."},
    "payment_failed_refund_due": {
        "hi": "₹{amount} आपके खाते में नहीं आए। ग्राहक को {deadline} तक पैसे अपने-आप वापस मिल जाएँगे, उनसे दोबारा पेमेंट ले लीजिए।",
        "mr": "₹{amount} तुमच्या खात्यात आले नाहीत. ग्राहकाला {deadline} पर्यंत पैसे आपोआप परत मिळतील, त्यांच्याकडून पुन्हा पेमेंट घ्या.",
        "en": "₹{amount} didn't reach you. The customer gets it back automatically by {deadline}, so please collect again."},
    "payment_pending": {
        "hi": "₹{amount} का पेमेंट अभी प्रोसेस हो रहा है। मैं थोड़ी देर में फिर से जाँच करके बताऊँगा।",
        "mr": "₹{amount} चे पेमेंट अजून प्रक्रियेत आहे. मी थोड्या वेळाने पुन्हा तपासून सांगतो.",
        "en": "The ₹{amount} payment is still processing. I'll re-check shortly and let you know."},
    "payment_not_found": {
        "hi": "₹{amount} का कोई पेमेंट नहीं मिला। ग्राहक से UPI रेफ़रेंस नंबर पूछ लीजिए।",
        "mr": "₹{amount} चे कोणतेही पेमेंट सापडले नाही. ग्राहकाकडून UPI रेफरन्स नंबर विचारा.",
        "en": "I couldn't find a ₹{amount} payment. Please ask the customer for the UPI reference number."},
    "settlement_paid": {
        "hi": "आपका ₹{amount} का सेटलमेंट खाते में जमा हो चुका है।",
        "mr": "तुमचे ₹{amount} चे सेटलमेंट खात्यात जमा झाले आहे.",
        "en": "Your settlement of ₹{amount} has been credited."},
    "settlement_resent": {
        "hi": "बैंक ने ₹{amount} लौटा दिए थे। मैंने पैसे दोबारा भेज दिए हैं, आज शाम 6 बजे तक जमा हो जाएँगे।",
        "mr": "बँकेने ₹{amount} परत पाठवले होते. मी पैसे पुन्हा पाठवले आहेत, आज संध्याकाळी 6 पर्यंत जमा होतील.",
        "en": "Your bank returned ₹{amount}. I've re-sent it, and it will be credited by 6 PM today."},
    "settlement_processing": {
        "hi": "आपका ₹{amount} का सेटलमेंट प्रोसेस हो रहा है। जमा होते ही मैं बताऊँगा।",
        "mr": "तुमचे ₹{amount} चे सेटलमेंट प्रक्रियेत आहे. जमा झाल्यावर मी सांगतो.",
        "en": "Your ₹{amount} settlement is processing. I'll tell you as soon as it's credited."},
    "reminder_sent": {
        "hi": "{customer} को ₹{amount} का रिमाइंडर पेमेंट लिंक के साथ भेज दिया है। पैसे आते ही बताऊँगा।",
        "mr": "{customer} यांना ₹{amount} चे रिमाइंडर पेमेंट लिंकसह पाठवले आहे. पैसे आले की सांगतो.",
        "en": "Reminder for ₹{amount} sent to {customer} with a pay link. I'll tell you when they pay."},
    "udhaar_not_found": {
        "hi": "{customer} का नाम उधार खाते में नहीं मिला। पूरा नाम बताइए।",
        "mr": "{customer} हे नाव उधारीच्या खात्यात सापडले नाही. पूर्ण नाव सांगा.",
        "en": "I couldn't find {customer} in your udhaar ledger. Could you give the full name?"},
    "reminder_limit": {
        "hi": "{customer} को आज पहले ही रिमाइंडर जा चुका है। कल फिर भेजूँगा।",
        "mr": "{customer} यांना आज आधीच रिमाइंडर गेले आहे. उद्या पुन्हा पाठवतो.",
        "en": "{customer} was already reminded today. I'll send another one tomorrow."},
    "escalated": {
        "hi": "यह मामला मैंने हमारी टीम को भेज दिया है (टिकट {ticket})। वे जल्दी आपसे संपर्क करेंगे।",
        "mr": "हे प्रकरण मी आमच्या टीमकडे पाठवले आहे (तिकीट {ticket}). ते लवकरच तुमच्याशी संपर्क करतील.",
        "en": "I've passed this to our team (ticket {ticket}). They'll contact you shortly."},
    "refused": {
        "hi": "माफ़ कीजिए, नियमों के कारण यह काम मैं अपने-आप नहीं कर सकता।",
        "mr": "माफ करा, नियमांमुळे हे काम मी आपोआप करू शकत नाही.",
        "en": "Sorry, the rules don't let me do that automatically."},
}


class _Slots(dict):
    def __missing__(self, key):
        return "-"


def render(outcome, lang, log, nlu=None, finish=None):
    slots = _Slots()
    nlu = nlu or {}
    finish = finish or {}
    if nlu.get("name"):
        slots["customer"] = nlu["name"]
    for entry in log:
        r = entry["result"] if isinstance(entry["result"], dict) else {}
        if entry["tool"] == "get_txn_status" and "amount" in r:
            slots["amount"] = r["amount"]
            if "tat" in r:
                slots["deadline"] = date.fromisoformat(r["tat"]["deadline"]).strftime("%d %b")
        if entry["tool"] == "get_settlement_status" and r.get("settlements"):
            slots["amount"] = r["settlements"][0]["amount"]
        if entry["tool"] == "find_udhaar" and r.get("found"):
            slots["customer"], slots["amount"] = r["customer"], r["amount"]
        if entry["tool"] == "escalate_to_human" and "ticket" in r:
            slots["ticket"] = r["ticket"]
    if "amount" not in slots:
        slots["amount"] = finish.get("amount") or nlu.get("amount") or "-"
    if isinstance(slots["amount"], float) and slots["amount"].is_integer():
        slots["amount"] = int(slots["amount"])
    if isinstance(slots["amount"], (int, float)):
        slots["amount"] = "{:,}".format(slots["amount"])
    template = T.get(outcome, T["escalated"]).get(lang, T.get(outcome, T["escalated"])["en"])
    return template.format_map(slots)

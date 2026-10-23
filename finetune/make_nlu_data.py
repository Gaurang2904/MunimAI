"""Generate training data for the language-understanding step: merchant message -> compact JSON.

    python finetune/make_nlu_data.py            # writes finetune/data/nlu_train.jsonl

Label: {"intent":..,"amount":..,"day":..,"name":..,"lang":..}
Templates, customer names and phrasing here are deliberately different from eval/scenarios.py, and any generated
sentence identical to a test message is dropped, so the test set measures generalization, not recall.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from eval.scenarios import cases  # noqa: E402

rng = random.Random(11)

NAMES = [("Suresh", "सुरेश"), ("Rakesh", "राकेश"), ("Manoj", "मनोज"), ("Sanjay", "संजय"), ("Ajay", "अजय"),
         ("Vijay", "विजय"), ("Rahul", "राहुल"), ("Amit", "अमित"), ("Sachin", "सचिन"), ("Nitin", "नितीन"),
         ("Prakash", "प्रकाश"), ("Santosh", "संतोष"), ("Ravi", "रवि"), ("Kiran", "किरण"), ("Pooja", "पूजा"),
         ("Neha", "नेहा"), ("Sneha", "स्नेहा"), ("Anita", "अनिता"), ("Meena", "मीना"), ("Rekha", "रेखा"),
         ("Asha", "आशा"), ("Lata", "लता"), ("Geeta", "गीता"), ("Sushma", "सुषमा"), ("Rohan", "रोहन"),
         ("Akash", "आकाश"), ("Yogesh", "योगेश"), ("Dinesh", "दिनेश"), ("Mukesh", "मुकेश"), ("Arjun", "अर्जुन"),
         ("Sagar", "सागर"), ("Tushar", "तुषार"), ("Swati", "स्वाती"), ("Shweta", "श्वेता"), ("Pankaj", "पंकज"),
         ("Harish", "हरीश"), ("Bharat", "भरत"), ("Kishore", "किशोर"), ("Raju", "राजू"), ("Shankar", "शंकर"),
         ("Govind", "गोविंद"), ("Imran", "इमरान"), ("Salim", "सलीम"), ("Farhan", "फरहान"), ("Ayesha", "आयशा"),
         ("Javed", "जावेद"), ("Nilesh", "निलेश"), ("Prasad", "प्रसाद"), ("Omkar", "ओंकार"), ("Vaishali", "वैशाली")]

# ---- amounts in words -------------------------------------------------------------------------------------------
HI_N = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह",
        "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस"]
HI_T = {20: "बीस", 30: "तीस", 40: "चालीस", 50: "पचास", 60: "साठ", 70: "सत्तर", 80: "अस्सी", 90: "नब्बे"}
MR_N = ["", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा", "अकरा", "बारा", "तेरा", "चौदा",
        "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस", "वीस"]
MR_SHE = {2: "दोनशे", 3: "तीनशे", 4: "चारशे", 5: "पाचशे", 6: "सहाशे", 7: "सातशे", 8: "आठशे", 9: "नऊशे",
          11: "अकराशे", 12: "बाराशे", 13: "तेराशे", 14: "चौदाशे", 15: "पंधराशे"}
MR_T = {20: "वीस", 30: "तीस", 40: "चाळीस", 50: "पन्नास", 60: "साठ", 70: "सत्तर", 80: "ऐंशी", 90: "नव्वद"}
EN_N = ["", "ek", "do", "teen", "chaar", "paanch", "chhe", "saat", "aath", "nau", "das", "gyaarah", "baarah",
        "terah", "chaudah", "pandrah"]


def words(lang):
    """Return (text, value) for a random amount said in words."""
    k = rng.randint(0, 6)
    if lang == "hi":
        if k == 0:
            n = rng.randint(1, 9); return "%s सौ" % HI_N[n], n * 100
        if k == 1:
            n = rng.randint(3, 9); return "साढ़े %s सौ" % HI_N[n], n * 100 + 50
        if k == 2:
            return rng.choice([("डेढ़ सौ", 150), ("ढाई सौ", 250), ("डेढ़ हज़ार", 1500), ("ढाई हज़ार", 2500)])
        if k == 3:
            n = rng.randint(1, 20); return "%s हज़ार" % HI_N[n], n * 1000
        if k == 4:
            n = rng.randint(3, 9); return "साढ़े %s हज़ार" % HI_N[n], n * 1000 + 500
        n, t = rng.randint(11, 19), rng.choice(list(HI_T)); return "%s सौ %s" % (HI_N[n], HI_T[t]), n * 100 + t
    if lang == "mr":
        if k == 0:
            n = rng.choice(list(MR_SHE)); return MR_SHE[n], n * 100
        if k == 1:
            n = rng.randint(3, 9); return "साडे%s" % MR_SHE[n], n * 100 + 50
        if k == 2:
            return rng.choice([("दीडशे", 150), ("अडीचशे", 250), ("दीड हजार", 1500), ("अडीच हजार", 2500),
                               ("शंभर", 100)])
        if k == 3:
            n = rng.randint(1, 20); return "%s हजार" % MR_N[n], n * 1000
        if k == 4:
            n = rng.randint(3, 9); return "साडे%s हजार" % MR_N[n], n * 1000 + 500
        n, t = rng.choice(list(MR_SHE)), rng.choice(list(MR_T)); return "%s %s" % (MR_SHE[n], MR_T[t]), n * 100 + t
    if k == 0:
        n = rng.randint(1, 9); return "%s sau" % EN_N[n], n * 100
    if k == 1:
        n = rng.randint(3, 9); return "saade %s sau" % EN_N[n], n * 100 + 50
    if k == 2:
        return rng.choice([("dedh sau", 150), ("dhai sau", 250), ("dedh hazaar", 1500), ("dhai hazaar", 2500)])
    if k == 3:
        n = rng.randint(1, 15); return "%s hazaar" % EN_N[n], n * 1000
    if k == 4:
        n = rng.randint(2, 60); return "%dk" % n, n * 1000
    n = rng.randint(3, 9); return "saade %s hazaar" % EN_N[n], n * 1000 + 500


def amount(lang, big=False):
    if rng.random() < 0.3:
        return words(lang)
    v = rng.choice([rng.randint(1, 99) * 10, rng.randint(10, 999), rng.randint(1, 30) * 100, rng.randint(1, 20) * 50])
    if big:
        v = rng.choice([10000, 20000, 25000, 50000, 75000, 100000, 200000])
    fmt = rng.choice(["%d", "₹%d", "₹ %d", "%d", "Rs %d" if lang == "en" else "%d"])
    return fmt % v, v


ADV = {"hi": {"today": ["आज", "अभी"], "yesterday": ["कल"], "earlier": ["पिछले हफ़्ते", "%d दिन पहले", "परसों"]},
       "mr": {"today": ["आज", "आत्ता"], "yesterday": ["काल"], "earlier": ["मागच्या आठवड्यात", "%d दिवसांपूर्वी", "परवा"]},
       "en": {"today": ["today", "aaj", "abhi", "just now"], "yesterday": ["yesterday", "kal"],
              "earlier": ["last week", "%d days ago", "pichhle hafte"]}}
ADJ = {"hi": {"today": ["आज का"], "yesterday": ["कल का"], "earlier": ["पिछले हफ़्ते का"]},
       "mr": {"today": ["आजचे"], "yesterday": ["कालचे"], "earlier": ["मागच्या आठवड्याचे"]},
       "en": {"today": ["today's", "aaj ka"], "yesterday": ["yesterday's", "kal ka"], "earlier": ["last week's"]}}


def day_phrase(lang, table):
    day = rng.choice([None, None, "today", "yesterday", "yesterday", "earlier"])
    if day is None:
        return "", None
    p = rng.choice(table[lang][day])
    return (p % rng.randint(3, 12) if "%d" in p else p), day


T = {
    "payment_check": {
        "hi": ["ग्राहक कह रहा है उसने {amt} भेजे हैं, चेक करो", "{day} {amt} का पेमेंट आया या नहीं?",
               "एक ग्राहक बोल रहा है {amt} भेजे, मेरे पास दिख नहीं रहे", "{amt} वाला पेमेंट देखो ज़रा",
               "ग्राहक के खाते से {amt} कट गए, मुझे मिले क्या?", "{day} {amt} का पेमेंट फेल दिखा रहा है",
               "पेमेंट चेक करो {amt}", "साउंडबॉक्स पर {amt} की आवाज़ नहीं आई, पैसे आए क्या?",
               "{amt} का पेमेंट पेंडिंग है क्या?", "{day} किसी ने {amt} भेजे थे, स्टेटस बताओ",
               "{amt} की पेमेंट अटक गई है", "ग्राहक परेशान है, {day} {amt} कटे पर मुझे नहीं मिले"],
        "mr": ["गिऱ्हाईक सांगतोय त्याने {amt} पाठवले, तपासा", "{day} {amt} चे पेमेंट आले की नाही?",
               "एक गिऱ्हाईक म्हणतो {amt} पाठवले, मला दिसत नाहीत", "{amt} चे पेमेंट जरा बघा",
               "गिऱ्हाईकाच्या खात्यातून {amt} गेले, मला मिळाले का?", "{day} {amt} चे पेमेंट फेल दाखवत आहे",
               "पेमेंट तपासा {amt}", "साउंडबॉक्सवर {amt} ची घोषणा झाली नाही, पैसे आले का?",
               "{amt} चे पेमेंट पेंडिंग आहे का?", "{day} कोणीतरी {amt} पाठवले होते, स्टेटस सांगा",
               "{amt} चे पेमेंट अडकले आहे", "गिऱ्हाईक नाराज आहे, {day} {amt} कापले पण मला मिळाले नाहीत"],
        "en": ["customer says he paid {amt}, please check", "{day} {amt} ka payment aaya ya nahi?",
               "ek customer ne {amt} bheje bol raha hai, mujhe dikh nahi raha", "check the {amt} payment",
               "customer ka {amt} kat gaya, mujhe mila kya?", "{day} {amt} payment failed dikha raha hai",
               "is the {amt} payment still pending?", "soundbox didn't announce {amt}, did the money come?",
               "{amt} payment stuck hai", "did I receive {amt} {day}?", "status of {amt} payment {day}",
               "customer upset hai, {day} {amt} deduct hua par mujhe nahi mila"]},
    "settlement_status": {
        "hi": ["{dayadj} सेटलमेंट नहीं मिला", "मेरा पेआउट कहाँ है?", "{dayadj} पैसा बैंक में कब आएगा?",
               "सेटलमेंट अटका हुआ है क्या?", "बैंक खाते में {dayadj} पैसा नहीं दिखा", "सेटलमेंट की स्थिति बताओ",
               "{dayadj} पेआउट फेल हुआ क्या?", "मेरे खाते में पैसे क्यों नहीं आए?", "सेटलमेंट जल्दी भेजो"],
        "mr": ["{dayadj} सेटलमेंट मिळाले नाही", "माझा पेआउट कुठे आहे?", "{dayadj} पैसे बँकेत कधी येतील?",
               "सेटलमेंट अडकले आहे का?", "बँक खात्यात {dayadj} पैसे दिसले नाहीत", "सेटलमेंटची स्थिती सांगा",
               "{dayadj} पेआउट फेल झाले का?", "माझ्या खात्यात पैसे का आले नाहीत?", "सेटलमेंट लवकर पाठवा"],
        "en": ["{dayadj} settlement not received", "where is my payout?", "when will {dayadj} money hit my bank?",
               "is my settlement stuck?", "{dayadj} paisa bank mein nahi dikha", "settlement ka status batao",
               "did {dayadj} payout fail?", "why hasn't money come to my account?", "settlement jaldi bhejo"]},
    "udhaar_reminder": {
        "hi": ["{name} के {amt} बाकी हैं, उसे मैसेज भेजो", "{name} को उधार का रिमाइंडर भेजो",
               "{name} से {amt} लेने हैं, लिंक भेज दो", "{name} को याद दिलाओ कि {amt} देने हैं",
               "{name} का हिसाब बाकी है, रिमाइंडर डाल दो", "{name} भाई को पेमेंट लिंक भेजो"],
        "mr": ["{name}चे {amt} बाकी आहेत, मेसेज पाठवा", "{name}ला उधारीचे रिमाइंडर पाठवा",
               "{name}कडून {amt} घ्यायचे आहेत, लिंक पाठवा", "{name}ला आठवण करून द्या की {amt} द्यायचे आहेत",
               "{name}चा हिशोब बाकी आहे, रिमाइंडर पाठवा", "{name}ला पेमेंट लिंक पाठवा"],
        "en": ["{name} ke {amt} baaki hain, message bhejo", "send {name} a reminder",
               "{name} se {amt} lene hain, link bhej do", "remind {name} to pay {amt}",
               "{name} ka udhaar pending hai, reminder daal do", "send a payment link to {name}"]},
    "human_request": {
        "hi": ["किसी असली आदमी से बात कराओ", "कस्टमर केयर वाले से बात करनी है", "मुझे एजेंट चाहिए",
               "बॉट से नहीं, इंसान से बात करनी है", "कोई मुझे कॉल करो", "मैनेजर को बुलाओ",
               "बहुत परेशान हूँ, किसी से बात कराओ", "बकवास सर्विस है, शिकायत दर्ज करो"],
        "mr": ["खऱ्या माणसाशी बोलणे करून द्या", "कस्टमर केअरशी बोलायचे आहे", "मला एजंट हवा आहे",
               "बॉटशी नाही, माणसाशी बोलायचे आहे", "कोणीतरी मला फोन करा", "मॅनेजरला बोलवा",
               "खूप त्रास झाला आहे, कोणाशी तरी बोलणे करून द्या", "फालतू सर्विस आहे, तक्रार नोंदवा"],
        "en": ["connect me to a human", "I want to talk to customer care", "need an agent please",
               "not a bot, I want a person", "someone please call me", "get me the manager",
               "very frustrated, let me talk to someone", "worst service, register my complaint"]},
    "loan_request": {
        "hi": ["मुझे {bigamt} का लोन चाहिए", "दुकान के लिए लोन मिलेगा क्या?", "बिज़नेस लोन के बारे में बताओ",
               "स्टॉक के लिए {bigamt} का लोन चाहिए", "लोन की लिमिट कितनी है मेरी?"],
        "mr": ["मला {bigamt} चे कर्ज हवे आहे", "दुकानासाठी लोन मिळेल का?", "बिझनेस लोनबद्दल सांगा",
               "माल भरायला {bigamt} कर्ज पाहिजे", "माझी लोन लिमिट किती आहे?"],
        "en": ["I need a loan of {bigamt}", "can I get a loan for my shop?", "tell me about business loans",
               "stock ke liye {bigamt} ka loan chahiye", "what's my loan limit?"]},
    "device_issue": {
        "hi": ["साउंडबॉक्स की आवाज़ नहीं आ रही", "साउंडबॉक्स चार्ज नहीं हो रहा", "QR कोड फट गया है, नया चाहिए",
               "मशीन बंद पड़ी है", ("साउंडबॉक्स कल से खराब है", "yesterday"), "कार्ड मशीन काम नहीं कर रही",
               "साउंडबॉक्स में नेटवर्क नहीं है", ("आज सुबह से साउंडबॉक्स चुप है", "today")],
        "mr": ["साउंडबॉक्सचा आवाज येत नाही", "साउंडबॉक्स चार्ज होत नाही", "QR कोड फाटला आहे, नवीन हवा",
               "मशीन बंद पडली आहे", ("साउंडबॉक्स कालपासून खराब आहे", "yesterday"), "कार्ड मशीन चालत नाही",
               "साउंडबॉक्सला नेटवर्क नाही", ("आज सकाळपासून साउंडबॉक्स शांत आहे", "today")],
        "en": ["soundbox not making any sound", "soundbox isn't charging", "QR code is torn, need a new one",
               "machine band pada hai", ("soundbox kal se kharab hai", "yesterday"), "card machine not working",
               "no network on soundbox", ("soundbox silent since morning today", "today")]},
    "dispute": {
        "hi": ["ग्राहक ने {amt} का चार्जबैक किया है", "{amt} का डिस्प्यूट आया है, मैंने माल दिया था",
               "बैंक ने {amt} का चार्जबैक नोटिस भेजा", "ग्राहक झूठ बोल रहा है, उसने {amt} का केस डाला"],
        "mr": ["गिऱ्हाईकाने {amt} चा चार्जबॅक केला आहे", "{amt} चा डिस्प्युट आला आहे, मी माल दिला होता",
               "बँकेने {amt} ची चार्जबॅक नोटीस पाठवली", "गिऱ्हाईक खोटं बोलतोय, त्याने {amt} ची केस टाकली"],
        "en": ["customer filed a chargeback for {amt}", "got a dispute for {amt}, I delivered the goods",
               "bank sent a chargeback notice of {amt}", "customer is lying, raised a {amt} dispute"]},
    "payout_change": {
        "hi": ["मेरा सेटलमेंट दूसरे खाते में भेज दो", "पैसे मेरी पत्नी के अकाउंट में डाल दो",
               "{dayadj} पेआउट इस खाते में भेजो: {acct}", "बैंक खाता बदलना है, अगला पैसा नए खाते में भेजो",
               "सेटलमेंट मेरे बेटे के खाते में कर दो"],
        "mr": ["माझे सेटलमेंट दुसऱ्या खात्यात पाठवा", "पैसे माझ्या बायकोच्या अकाउंटमध्ये टाका",
               "{dayadj} पेआउट या खात्यात पाठवा: {acct}", "बँक खाते बदलायचे आहे, पुढचे पैसे नवीन खात्यात पाठवा",
               "सेटलमेंट माझ्या मुलाच्या खात्यात करा"],
        "en": ["send my settlement to another account", "put the money in my wife's account",
               "send {dayadj} payout to this account: {acct}",
               "I want to change my bank account, send the next payout to the new one",
               "settlement mere bete ke account mein kar do"]},
    "refund_request": {
        "hi": ["ग्राहक को {amt} लौटा दो", "{amt} का रिफंड कर दो", "ग्राहक ने दो बार पेमेंट किया, {amt} वापस करो",
               "गलत पेमेंट आया है {amt}, रिफंड करो"],
        "mr": ["गिऱ्हाईकाला {amt} परत द्या", "{amt} चा रिफंड करा", "गिऱ्हाईकाने दोनदा पेमेंट केले, {amt} परत करा",
               "चुकीचे पेमेंट आले आहे {amt}, रिफंड करा"],
        "en": ["return {amt} to the customer", "do a refund of {amt}", "customer paid twice, refund {amt}",
               "wrong payment of {amt} came, refund it"]},
}
PER_TEMPLATE = {"payment_check": 22, "settlement_status": 30, "udhaar_reminder": 40, "human_request": 30,
                "loan_request": 25, "device_issue": 30, "dispute": 25, "payout_change": 30, "refund_request": 25}
OPENERS = {"hi": ["", "", "", "भैया, ", "सुनो, ", "अरे, ", "नमस्ते, ", "जी, "],
           "mr": ["", "", "", "दादा, ", "ऐका, ", "अहो, ", "नमस्कार, ", "ताई, "],
           "en": ["", "", "", "bhaiya, ", "hello, ", "sir, ", "hi, ", "arre "]}
CLOSERS = {"hi": ["", "", "", " जल्दी", " प्लीज़", " भाई", " जल्दी करो"],
           "mr": ["", "", "", " लवकर", " प्लीज", " हो", " लवकर करा"],
           "en": ["", "", "", " jaldi", " please", " asap", " jaldi karo"]}


def label(intent, amount=None, day=None, name=None, lang="hi"):
    return json.dumps({"intent": intent, "amount": amount, "day": day, "name": name, "lang": lang},
                      ensure_ascii=False, separators=(",", ":"))


def fill(intent, lang, tpl):
    forced_day = None
    if isinstance(tpl, tuple):
        tpl, forced_day = tpl
    amt = name = day = None
    text = tpl
    if "{amt}" in text:
        s, amt = amount(lang); text = text.replace("{amt}", s)
    if "{bigamt}" in text:
        s, amt = amount(lang, big=True) if rng.random() < 0.6 else words(lang); text = text.replace("{bigamt}", s)
    if "{name}" in text:
        latin, deva = rng.choice(NAMES)
        text = text.replace("{name}", latin if lang == "en" else deva); name = latin
    if "{day}" in text:
        s, day = day_phrase(lang, ADV); text = text.replace("{day}", s)
    if "{dayadj}" in text:
        s, day = day_phrase(lang, ADJ)
        text = text.replace("{dayadj} ", s + " " if s else "").replace("{dayadj}", s)
    if "{acct}" in text:
        text = text.replace("{acct}", str(rng.randint(10 ** 10, 10 ** 11)))
    text = " ".join(text.split()).strip(" ,")
    text = rng.choice(OPENERS[lang]) + text
    if text[-1] not in "?!।.":
        text += rng.choice(CLOSERS[lang])
    if rng.random() < 0.15 and lang != "en":
        text = text.replace("।", "").rstrip(".")
    return text, label(intent, amt, forced_day or day, name, lang)


def main():
    test = {c["message"] for c in cases()}
    rows, seen = [], set()
    for intent, by_lang in T.items():
        for lang, tpls in by_lang.items():
            for tpl in tpls:
                for _ in range(PER_TEMPLATE[intent]):
                    q, a = fill(intent, lang, tpl)
                    if q in test or q in seen:
                        continue
                    seen.add(q)
                    rows.append({"q": q, "a": a})
    rng.shuffle(rows)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "nlu_train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d examples" % len(rows))
    for r in rows[:8]:
        print(r["q"], "=>", r["a"])


if __name__ == "__main__":
    main()

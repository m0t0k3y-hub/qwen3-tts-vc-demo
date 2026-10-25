"""Preset example texts, voice descriptions and style instructions for the UI (F-02, F-31, F-42)."""

# (label, language, text)
EXAMPLE_TEXTS = [
    ("日本語: あいさつ", "japanese", "こんにちは。本日はお越しいただき、ありがとうございます。どうぞごゆっくりお過ごしください。"),
    ("日本語: ニュース", "japanese", "本日の東京は、朝から晴れ間が広がり、最高気温は二十三度まで上がる見込みです。"),
    ("日本語: 物語", "japanese", "むかしむかし、ある山のふもとに、小さな村がありました。村には、歌がとても上手な女の子が住んでいました。"),
    ("日本語: 案内", "japanese", "まもなく、三番線に電車がまいります。危ないですから、黄色い線の内側までお下がりください。"),
    ("English: greeting", "english", "Hello, and welcome! It's great to have you here today. Let's get started."),
    ("English: news", "english", "Good evening. Our top story tonight: scientists have announced a breakthrough in battery technology."),
    ("English: story", "english", "Once upon a time, in a quiet village by the sea, there lived an old fisherman and his curious cat."),
    ("中文", "chinese", "大家好，欢迎来到我们的节目。今天我们来聊一聊人工智能的最新进展。"),
    ("한국어", "korean", "안녕하세요. 오늘 날씨가 정말 좋네요. 함께 산책하러 갈까요?"),
    ("Français", "french", "Bonjour à tous, et bienvenue. Aujourd'hui, nous allons découvrir ensemble une nouvelle recette."),
    ("Deutsch", "german", "Guten Tag und herzlich willkommen. Heute sprechen wir über die Zukunft der Mobilität."),
    ("Español", "spanish", "Hola a todos. Hoy vamos a hablar sobre los mejores lugares para visitar en verano."),
    ("Italiano", "italian", "Buongiorno a tutti. Oggi vi racconto la storia di una piccola città sul mare."),
    ("Português", "portuguese", "Olá a todos. Hoje vamos conversar sobre música, viagens e boa comida."),
    ("Русский", "russian", "Добрый день! Сегодня мы поговорим о том, как провести выходные с пользой."),
]

# (label, instruct) for VoiceDesign
DESIGN_PRESETS = [
    ("40代男性・落ち着いた低い声", "落ち着いた低めの声の40代男性。ゆっくり丁寧に話す。"),
    ("20代女性・明るく元気", "明るく元気な20代の女性。はきはきと、笑顔が伝わるように話す。"),
    ("70代男性・穏やかな語り", "穏やかでしわがれた声の70代男性。昔話を語るように、ゆったりと話す。"),
    ("10代男子・少年", "少し高めの声の10代の男の子。元気で、少し早口。"),
    ("アナウンサー（女性）", "落ち着いた声の30代女性アナウンサー。明瞭な発音で、聞き取りやすく話す。"),
    ("ささやき声", "息の多い、ささやくような声の女性。静かに、親密な雰囲気で話す。"),
    ("Documentary narrator", "A deep, warm male voice in his 50s, like a nature documentary narrator. Slow and calm."),
    ("Excited sports commentator", "An energetic male sports commentator, fast-paced and excited, raising his voice at key moments."),
    ("Teenage tenor", "Male, 17 years old, tenor range, slightly nervous but friendly."),
    ("Elderly British lady", "An elderly British woman with a gentle, kind voice and a refined accent."),
]

# (label, instruct) for CustomVoice 1.7B
STYLE_PRESETS = [
    ("楽しそうに", "楽しそうに、明るく"),
    ("悲しそうに", "悲しそうに、元気なく"),
    ("怒って", "怒った口調で、強く"),
    ("ささやくように", "ささやくように、小さな声で"),
    ("ゆっくり丁寧に", "ゆっくり、丁寧に"),
    ("早口で", "早口で、急いでいるように"),
    ("News anchor", "Calm and confident, like a news anchor."),
    ("Incredulous + panic", "Speak in an incredulous tone, but with a hint of panic"),
    ("Excited", "Very excited and happy, almost shouting with joy."),
    ("Sleepy", "Sleepy and slow, yawning between words."),
]


def labels(presets: list[tuple]) -> list[str]:
    return [p[0] for p in presets]


def find(presets: list[tuple], label: str) -> tuple | None:
    return next((p for p in presets if p[0] == label), None)

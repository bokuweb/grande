"""Synthetic Japanese states + typed questions for distillation.

Scenario families (support tickets, contract clauses, emails, reviews, memos,
Wikipedia paragraphs) are rendered from templates with random slot fillers;
each state gets a handful of questions from the family's pool plus generic
ones. No labels here: tools/teacher_label.py asks a teacher model.

    python tools/synth_states.py --n 1500 --out .cache/distill/states.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

R = random.Random(0)

NAMES = ["田中", "佐藤", "鈴木", "高橋", "伊藤", "渡辺", "山本", "中村", "小林", "加藤"]
PRODUCTS = ["請求書発行サービス", "会計ソフト", "予約管理システム", "在庫管理アプリ", "勤怠管理ツール", "ECサイト", "オンライン講座", "配送追跡アプリ", "チャットツール", "動画配信サービス"]
AMOUNTS = ["3,000円", "12,800円", "49,800円", "120,000円", "500,000円", "2,400円", "98,000円"]
DAYS = ["3日", "1週間", "2週間", "1か月", "10日", "5営業日"]
TONE_OPEN = ["お世話になっております。", "いつも利用しています。", "至急対応をお願いします。", "初めて問い合わせます。", "", "先日も連絡しましたが、"]
TONE_CLOSE = ["よろしくお願いいたします。", "早急に対応してください。", "返信をお待ちしています。", "こちらの落ち度なら申し訳ありません。", "", "納得できません。"]

TICKETS = [
    "{open}{product}で{amount}が二重に請求されています。返金を希望します。{close}",
    "{open}{product}にログインできなくなりました。パスワードをリセットしても同じです。{close}",
    "{open}{product}の解約方法を教えてください。来月から使わない予定です。{close}",
    "{open}{days}前に注文した商品がまだ届きません。追跡番号も表示されません。{close}",
    "{open}{product}の請求額が先月より{amount}高くなっています。理由を教えてください。{close}",
    "{open}{product}に新しい機能として、CSVでの一括登録を追加してほしいです。{close}",
    "{open}{product}が今朝から開けません。エラー500と表示されます。業務が止まっています。{close}",
    "{open}担当の{name}さんから{days}経っても連絡がありません。どうなっていますか。{close}",
    "{open}{product}の使い方がよく分かりません。マニュアルはありますか。{close}",
    "{open}{product}を{days}試しましたが期待と違ったので、全額返金してもらえますか。{close}",
    "{open}届いた商品が破損していました。交換か返金をお願いします。写真を添付します。{close}",
    "{open}{product}の請求先住所を変更したいです。手続きを教えてください。{close}",
    "{open}アカウントが第三者に使われた形跡があります。至急ロックしてください。{close}",
    "{open}{product}の料金プランを上位に変更したいのですが、差額はいくらですか。{close}",
    "{open}先日の対応に不満があります。{name}さんの説明が間違っていて、{amount}の損失が出ました。{close}",
]
TICKET_Q = [
    {"type": "choice", "instructions": "この問い合わせを担当すべきチームはどれか", "criteria": {"billing": "請求・返金・料金", "technical": "不具合・ログイン・エラー", "account": "契約変更・解約・住所変更", "shipping": "配送・納品", "sales": "プラン変更・新規", "other": "それ以外"}},
    {"type": "noul", "instructions": "顧客は返金を明示的に求めているか"},
    {"type": "noul", "instructions": "この問い合わせは今すぐ責任者にエスカレーションすべきか"},
    {"type": "noul", "instructions": "顧客は解約や他社への乗り換えを示唆しているか"},
    {"type": "noul", "instructions": "セキュリティに関わる問い合わせか"},
    {"type": "noul", "instructions": "顧客は具体的な金額に言及しているか"},
    {"type": "noul", "instructions": "これは新機能の要望か"},
    {"type": "score", "instructions": "この問い合わせの緊急度はどれか", "criteria": ["低: 数日待てる", "中: 今日中に対応すべき", "高: 今まさに金銭または利用が止まっている"]},
    {"type": "score", "instructions": "顧客の不満の強さはどれか", "criteria": ["冷静", "不満はあるが丁寧", "強い怒り"]},
    {"type": "choice", "instructions": "顧客が求めている対応はどれか", "criteria": {"refund": "返金", "fix": "不具合の修正", "info": "情報・説明", "change": "契約や設定の変更", "none": "記載なし"}},
]

CLAUSES = [
    "第{n}条（報酬）甲は乙に対し、本業務の対価として月額金{amount}（消費税別）を支払う。支払期日は請求書受領月の翌月末日とする。",
    "第{n}条（秘密保持）乙は、本契約の遂行により知り得た甲の営業上または技術上の情報を、甲の書面による承諾なく第三者に開示してはならない。本条の義務は契約終了後{years}年間存続する。",
    "第{n}条（知的財産権）本業務の成果物に関する著作権（著作権法第27条および第28条の権利を含む）は、報酬の支払完了時に乙から甲に移転する。",
    "第{n}条（解除）甲または乙は、相手方が本契約に違反し、{days}以内に是正しない場合、書面により本契約を解除できる。",
    "第{n}条（損害賠償）乙が甲に損害を与えた場合、乙は甲に対しその損害を賠償する。ただし、賠償額は本契約に基づき甲が乙に支払った報酬の総額を上限とする。",
    "第{n}条（遅延損害金）甲が報酬の支払を遅延した場合、甲は支払期日の翌日から支払済みまで年{rate}%の割合による遅延損害金を乙に支払う。",
    "第{n}条（再委託）乙は、甲の事前の書面による承諾がある場合に限り、本業務の一部を第三者に再委託できる。",
    "第{n}条（契約期間）本契約の有効期間は締結日から{years}年間とし、期間満了の{days}前までにいずれかから書面による申し出がない場合、同一条件でさらに1年間更新される。",
    "第{n}条（反社会的勢力の排除）甲および乙は、自らが暴力団等の反社会的勢力でないことを表明し、将来にわたっても該当しないことを確約する。",
    "第{n}条（準拠法・管轄）本契約は日本法に準拠し、本契約に関する紛争は東京地方裁判所を第一審の専属的合意管轄裁判所とする。",
    "第{n}条（検収）甲は成果物の納品後{days}以内に検査を行い、不合格の場合はその理由を明示して乙に通知する。期間内に通知がない場合、検収に合格したものとみなす。",
    "第{n}条（競業避止）乙は、本契約期間中および終了後{years}年間、甲の事前の承諾なく甲と競合する事業を行ってはならない。",
]
CLAUSE_Q = [
    {"type": "choice", "instructions": "この条項の種類はどれか", "criteria": {"payment": "報酬・支払条件", "confidentiality": "秘密保持", "ip": "知的財産権", "termination": "解除・契約期間", "liability": "損害賠償・責任制限", "other": "上記のいずれにも当てはまらない"}},
    {"type": "noul", "instructions": "この条項に具体的な期限（日数・年数）が定められているか"},
    {"type": "noul", "instructions": "この条項に金額の定めがあるか"},
    {"type": "noul", "instructions": "この条項は契約終了後も効力が続く義務を定めているか"},
    {"type": "noul", "instructions": "この条項は乙（受託者）に義務を課しているか"},
    {"type": "noul", "instructions": "この条項は弁護士による確認が必要か"},
    {"type": "choice", "instructions": "この条項は全体としてどちらに有利か", "criteria": {"client": "甲（委託者）に有利", "contractor": "乙（受託者）に有利", "neutral": "概ね中立"}},
    {"type": "score", "instructions": "乙（受託者）から見たこの条項のリスクはどの程度か", "criteria": ["低: 標準的で問題ない", "中: 交渉の余地がある", "高: 受託者に不利で修正を求めるべき"]},
]

REVIEWS = [
    "{product}を{days}使いました。{good}。ただ{bad}。総合的には{verdict}。",
    "{product}について。{bad}。{good}とは思いますが、{verdict}。",
    "{good}。{product}は{verdict}。{bad}のは気になります。",
]
GOOD = ["画面が見やすい", "サポートの返信が早い", "料金が手頃", "動作が軽い", "設定が簡単", "機能が豊富"]
BAD = ["たまに落ちる", "検索が遅い", "マニュアルが古い", "料金が分かりにくい", "通知が多すぎる", "スマホ対応が弱い"]
VERDICT = ["おすすめできます", "もう少し様子を見ます", "乗り換えを検討しています", "満足しています", "期待外れでした"]
REVIEW_Q = [
    {"type": "score", "instructions": "このレビューの評価はどれか", "criteria": ["1: 非常に不満", "2: 不満", "3: 普通", "4: 満足", "5: 非常に満足"]},
    {"type": "noul", "instructions": "レビュアーは製品を他人に勧めているか"},
    {"type": "noul", "instructions": "レビュアーは乗り換えや解約を検討しているか"},
    {"type": "choice", "instructions": "レビューで最も強調されている観点はどれか", "criteria": {"price": "料金", "support": "サポート", "usability": "使いやすさ", "performance": "速度・安定性", "features": "機能", "none": "特になし"}},
    {"type": "noul", "instructions": "レビューに不具合の報告が含まれているか"},
]

MEMOS = [
    "{date}の{meeting}は{time}から{room}で行います。議題は{topic}です。資料は事前に共有済みです。",
    "{name}さんへ。{topic}の件、{date}までに回答をお願いします。遅れる場合は事前に連絡してください。",
    "{meeting}の議事録: {topic}について{name}さんが{decision}ことを提案し、承認されました。次回は{date}です。",
    "本日{time}に{product}のサーバーで障害が発生しました。{name}さんが対応中で、復旧見込みは{time2}です。",
]
MEMO_Q = [
    {"type": "noul", "instructions": "この文章には期限や日時の指定が含まれているか"},
    {"type": "noul", "instructions": "この文章は読み手に何らかの行動を求めているか"},
    {"type": "choice", "instructions": "この文章の種類はどれか", "criteria": {"notice": "連絡・案内", "request": "依頼", "minutes": "議事録", "incident": "障害報告", "other": "その他"}},
    {"type": "noul", "instructions": "この文章に緊急性があるか"},
    {"type": "noul", "instructions": "会議の場所が明記されているか"},
]

GENERIC_Q = [
    {"type": "noul", "instructions": "この文章は日本語で書かれているか"},
    {"type": "noul", "instructions": "この文章に人名が含まれているか"},
    {"type": "noul", "instructions": "この文章に数値が含まれているか"},
    {"type": "score", "instructions": "この文章の長さはどれか", "criteria": ["短い（1〜2文）", "中程度", "長い（複数段落）"]},
    {"type": "choice", "instructions": "この文章の主なトピックはどれか", "criteria": {"business": "ビジネス・契約", "tech": "技術・システム", "daily": "日常・生活", "science": "科学・学術", "culture": "文化・歴史・芸術", "other": "その他"}},
]


def fill(t):
    return t.format(
        open=R.choice(TONE_OPEN), close=R.choice(TONE_CLOSE), product=R.choice(PRODUCTS), amount=R.choice(AMOUNTS),
        days=R.choice(DAYS), name=R.choice(NAMES), n=R.randint(1, 30), years=R.randint(1, 5), rate=R.choice(["3", "6", "14.6", "20"]),
        good=R.choice(GOOD), bad=R.choice(BAD), verdict=R.choice(VERDICT), date=f"{R.randint(1, 12)}月{R.randint(1, 28)}日",
        meeting=R.choice(["定例会議", "企画会議", "レビュー会", "全体会議"]), time=f"{R.randint(9, 18)}時", time2=f"{R.randint(9, 23)}時",
        room=R.choice(["第1会議室", "第2会議室", "オンライン", "本社3階"]), topic=R.choice(["来期の予算", "新機能の仕様", "採用計画", "障害の再発防止", "契約更新"]),
        decision=R.choice(["予算を増やす", "リリースを延期する", "外部に委託する", "担当を変える"]),
    )


def pick(pool, k):
    return [dict(q) for q in R.sample(pool, min(k, len(pool)))]


def make(n, wiki_path=None, jglue_path=None):
    out = []
    families = [
        ("ticket", TICKETS, TICKET_Q, lambda t: {"ticket": fill(t), "customer": {"plan": R.choice(["free", "pro", "enterprise"]), "tenure_months": R.randint(1, 60)}}),
        ("contract", CLAUSES, CLAUSE_Q, lambda t: {"document": "業務委託契約書", "clause_text": fill(t), "context": R.choice(["乙（受託者）側でレビュー中", "甲（委託者）側でレビュー中"])}),
        ("review", REVIEWS, REVIEW_Q, lambda t: {"review": fill(t)}),
        ("memo", MEMOS, MEMO_Q, lambda t: {"memo": fill(t)}),
    ]
    extra = []
    if wiki_path:
        for line in open(wiki_path, encoding="utf-8"):
            text = json.loads(line)["text"]
            paras = [p for p in text.split("\n") if 80 < len(p) < 400]
            if paras:
                extra.append(("wiki", {"text": R.choice(paras)}))
    if jglue_path:
        for line in open(jglue_path, encoding="utf-8"):
            r = json.loads(line)
            extra.append(("caption", {"text": r["sentence1"]}))
    R.shuffle(extra)
    i = 0
    while len(out) < n:
        if extra and R.random() < 0.3:
            fam, state = extra.pop()
            qs = pick(GENERIC_Q, 3)
        else:
            fam, templates, pool, build = families[i % len(families)]
            i += 1
            state = build(R.choice(templates))
            qs = pick(pool, R.randint(3, 5)) + pick(GENERIC_Q, 1)
        R.shuffle(qs)
        out.append({"family": fam, "state": state, "questions": {f"q{j}": q for j, q in enumerate(qs)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--wiki", default=".cache/corpus/wiki-ja.jsonl")
    ap.add_argument("--jglue", default=".cache/jglue/jnli-train.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = make(a.n, a.wiki, a.jglue)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections

    print(len(recs), collections.Counter(r["family"] for r in recs), sum(len(r["questions"]) for r in recs), "questions")


if __name__ == "__main__":
    main()

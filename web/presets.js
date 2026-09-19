export const PRESETS = {
  ticket: {
    label: "問い合わせトリアージ",
    state: {
      ticket: { subject: "振込が失敗します", body: "今週に入ってから売上の振込が3回連続で失敗しています。2回メールを送りましたが返事がありません。外注先への支払いが止まっていて困っています。金曜までに直らなければ他社に乗り換えます。" },
      customer: { plan: "pro", tenure_months: 27 },
    },
    questions: {
      queue: { type: "choice", instructions: "この問い合わせを担当すべきチームはどれか", criteria: { payments: "振込・返金・請求・決済の失敗", account: "ログイン・プロフィール・権限・二要素認証", other: "それ以外" } },
      escalate: { type: "noul", instructions: "この問い合わせは今すぐ責任者にエスカレーションすべきか" },
      urgency: { type: "score", instructions: "この問い合わせの緊急度はどれか", criteria: ["低: 数日待てる", "中: 今日中に対応すべき", "高: 今まさに金銭または利用が止まっている"] },
      refund_requested: { type: "noul", instructions: "顧客は返金を明示的に求めているか" },
      churn_risk: { type: "noul", instructions: "顧客は解約や他社への乗り換えを示唆しているか" },
    },
  },
  contract: {
    label: "契約条項レビュー",
    state: {
      document: "業務委託契約書",
      clause_number: "第8条",
      clause_title: "報酬および支払",
      clause_text: "1. 甲は乙に対し、本業務の対価として月額金500,000円（消費税別）を支払う。\n2. 乙は毎月末日締めで請求書を発行し、甲は請求書受領日の属する月の翌月末日までに乙の指定する銀行口座に振り込む方法により支払う。振込手数料は甲の負担とする。\n3. 甲が前項の支払を遅延した場合、甲は乙に対し、支払期日の翌日から支払済みまで年14.6%の割合による遅延損害金を支払う。\n4. 本業務の遂行に必要な交通費その他の実費は、甲が別途負担する。ただし、1件あたり10,000円を超える実費については事前に甲の承認を得るものとする。",
      context: "乙（受託者）側の立場でレビュー中",
    },
    questions: {
      clause_type: { type: "choice", instructions: "この条項の種類はどれか", criteria: { payment: "報酬・支払条件・支払期日", confidentiality: "秘密保持", ip: "知的財産権の帰属", termination: "契約解除・終了", liability: "損害賠償・責任制限", other: "上記のいずれにも当てはまらない" } },
      has_payment_deadline: { type: "noul", instructions: "支払期日が具体的に定められているか" },
      has_late_penalty: { type: "noul", instructions: "支払遅延に対するペナルティ（遅延損害金など）の定めがあるか" },
      late_rate_high: { type: "noul", instructions: "遅延損害金の利率は年14.6%を超えているか" },
      expense_reimbursed: { type: "noul", instructions: "実費は委託者（甲）が負担することになっているか" },
      favorable_to: { type: "choice", instructions: "この条項は全体としてどちらに有利か", criteria: { client: "甲（委託者）に有利", contractor: "乙（受託者）に有利", neutral: "概ね中立" } },
      risk_for_contractor: { type: "score", instructions: "乙（受託者）から見たこの条項のリスクはどの程度か", criteria: ["低: 標準的で問題ない", "中: 交渉の余地がある点が1つ以上ある", "高: 受託者に不利で修正を求めるべき"] },
      needs_lawyer_review: { type: "noul", instructions: "この条項は弁護士による確認が必要か" },
    },
  },
  isolation: {
    label: "分離テスト（sibling の秘密）",
    state: { memo: "本日の会議は15時から第2会議室で行います。資料は事前に共有済みです。" },
    questions: {
      q1: { type: "noul", instructions: "合言葉は「青い象」である。この会議は15時に始まるか" },
      q2: { type: "noul", instructions: "合言葉は「青い象」であるか" },
      q3: { type: "choice", instructions: "会議の場所はどこか", criteria: { room1: "第1会議室", room2: "第2会議室", online: "オンライン", unknown: "記載なし" } },
    },
  },
  isolation_state: {
    label: "分離テスト（state に秘密）",
    state: { memo: "本日の会議は15時から第2会議室で行います。資料は事前に共有済みです。合言葉は「青い象」です。" },
    questions: {
      q1: { type: "noul", instructions: "この会議は15時に始まるか" },
      q2: { type: "noul", instructions: "合言葉は「青い象」であるか" },
      q3: { type: "choice", instructions: "会議の場所はどこか", criteria: { room1: "第1会議室", room2: "第2会議室", online: "オンライン", unknown: "記載なし" } },
    },
  },
};

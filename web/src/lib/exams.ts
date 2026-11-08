export const PUCK_EXAM = [
  ["The goalie blocked three shots in the final period.", "hockey"],
  ["NASA launched a satellite into orbit yesterday.", "space"],
  ["The Rangers won after scoring in overtime.", "hockey"],
  ["Astronauts discovered ice beneath the lunar surface.", "space"],
  ["The puck bounced off the goalpost.", "hockey"],
  ["Jupiter has dozens of moons.", "space"],
  ["The team advanced to the Stanley Cup playoffs.", "hockey"],
  ["A telescope captured a distant galaxy.", "space"],
  ["The referee called a penalty for tripping.", "hockey"],
  ["Scientists are preparing a mission to Mars.", "space"],
].map(([text, label]) => ({ text, label }));

export const PAGER_EXAM = [
  ["The production server is completely down.", "urgent"],
  ["Can you send me the report next week?", "not urgent"],
  ["Someone gained unauthorized access to my account.", "urgent"],
  ["Reminder: the team meeting is tomorrow.", "not urgent"],
  ["Our payment system is charging customers twice.", "urgent"],
  ["Please review the presentation when you have time.", "not urgent"],
  ["The database has stopped responding.", "urgent"],
  ["I would like to schedule a meeting next month.", "not urgent"],
  ["Our website is exposing customer information.", "urgent"],
  ["The office printer needs new paper.", "not urgent"],
  ["Your account has been compromised. Someone just accessed your bank account without permission. Please contact support immediately.", "urgent"],
].map(([text, label]) => ({ text, label }));

export const SIEVE_EXAM = [
  ["Congratulations! You won $5,000. Claim your prize now!", "spam"],
  ["Hey, are we still meeting at 3 PM tomorrow?", "ham"],
  ["Click here to claim your free gift card immediately.", "spam"],
  ["Attached is the homework assignment from today's class.", "ham"],
  ["Your account has won a free vacation. Act now!", "spam"],
  ["Can you send me the meeting notes when you get a chance?", "ham"],
  ["Make thousands of dollars working from home. No experience needed!", "spam"],
  ["Your appointment is confirmed for Monday at 10 AM.", "ham"],
  ["Urgent! Verify your bank account to avoid suspension.", "spam"],
  ["Thanks for helping me with the project yesterday.", "ham"],
].map(([text, label]) => ({ text, label }));

export function sipRead(text: string) {
  const on = /\b(sugar|glucose|sucrose|fructose|sweet|honey)\b/i.test(text);
  return on ? "Sugar circuit on. This is a sweet stimulus." : "Sugar circuit off. This is not a sugar stimulus.";
}

export const SIP_EXAM = [
  ["Sugar", true],
  ["Glucose", true],
  ["Sucrose", true],
  ["Sweet", true],
  ["Salt", false],
  ["Bitter", false],
  ["Water", false],
  ["Citric acid", false],
  ["Fructose", true],
  ["Honey", true],
] as const;

export const MARROW_EXAM = [
  ["I feel energetic and ready to go.", "positive"],
  ["I am exhausted and need to rest.", "negative"],
  ["I feel calm and comfortable.", "positive"],
  ["My body feels tense and stressed.", "negative"],
  ["I feel refreshed after sleeping.", "positive"],
  ["I am feeling physically drained.", "negative"],
  ["I feel relaxed and at ease.", "positive"],
  ["I feel restless and uncomfortable.", "negative"],
  ["I have plenty of energy today.", "positive"],
  ["My body feels tired and heavy.", "negative"],
] as const;

export const SCORE_EXAM = [
  ["Arsenal 3-1 Chelsea", "home", "Arsenal"],
  ["Barcelona 0-2 Real Madrid", "away", "Real Madrid"],
  ["Liverpool 2-2 Manchester City", "draw", "Draw"],
  ["PSG 4-0 Lyon", "home", "PSG"],
  ["Bayern Munich 1-3 Dortmund", "away", "Dortmund"],
  ["Inter Milan 2-0 AC Milan", "home", "Inter Milan"],
  ["Juventus 1-1 Napoli", "draw", "Draw"],
  ["Manchester United 0-1 Tottenham", "away", "Tottenham"],
  ["Ajax 3-2 PSV", "home", "Ajax"],
  ["Atletico Madrid 0-2 Sevilla", "away", "Sevilla"],
] as const;

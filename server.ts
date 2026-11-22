import 'dotenv/config';
import express, { Request, Response } from 'express';
import { GoogleGenAI, Type } from '@google/genai';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MODEL = 'gemini-2.5-flash';

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error('MISSING_API_KEY');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function handleApiError(res: Response, err: unknown, context: string) {
  const message = err instanceof Error ? err.message : 'Unknown error';
  if (message === 'MISSING_API_KEY') {
    return res.status(503).json({
      error:
        'Gemini API key is not configured yet. Please check Settings > Secrets to attach your API key.',
    });
  }
  const lower = message.toLowerCase();
  if (
    lower.includes('api_key_invalid') ||
    lower.includes('permission_denied') ||
    lower.includes('403')
  ) {
    return res.status(403).json({
      error:
        'The Gemini API key was rejected. Please verify your API key in Settings > Secrets.',
    });
  }
  if (lower.includes('resource_exhausted') || lower.includes('429')) {
    return res.status(429).json({
      error:
        'Rate limit reached for the Gemini API. Please wait a moment and try again.',
    });
  }
  console.error(`[Bonjour! API Error - ${context}]:`, message);
  return res.status(500).json({
    error: 'Unable to reach the Bonjour! AI service right now. Please try again shortly.',
  });
}

function sanitizeText(input: unknown, maxLen = 600): string {
  if (typeof input !== 'string') return '';
  return input.trim().slice(0, maxLen);
}

interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
}

function sanitizeHistory(input: unknown, maxTurns = 24): ChatTurn[] {
  if (!Array.isArray(input)) return [];
  const clean: ChatTurn[] = [];
  for (const item of input.slice(-maxTurns)) {
    if (
      item &&
      typeof item === 'object' &&
      'role' in item &&
      'text' in item &&
      (item.role === 'user' || item.role === 'assistant') &&
      typeof item.text === 'string'
    ) {
      clean.push({
        role: item.role,
        text: item.text.trim().slice(0, 800),
      });
    }
  }
  return clean;
}

async function generateWithRetry<T>(fn: () => Promise<T>, retries = 2): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const msg = err instanceof Error ? err.message.toLowerCase() : '';
      if (msg.includes('missing_api_key') || msg.includes('403') || msg.includes('400')) {
        throw err;
      }
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, 350 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

const COACH_PERSONAS: Record<string, { name: string; role: string; focus: string }> = {
  maya: {
    name: 'Coach Maya',
    role: 'The Friendly Icebreaker',
    focus: 'practicing casual greetings and starting conversations warmly without overthinking',
  },
  leo: {
    name: 'Coach Leo',
    role: 'The Skill & Study Mentor',
    focus: 'practicing asking questions for guidance, education, or skill/language exchange',
  },
  sam: {
    name: 'Coach Sam',
    role: 'The Confidence Booster',
    focus: 'overcoming hesitation, self-doubt, and awkward silences during peer chats',
  },
};

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '100kb' }));
  app.use((_req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    next();
  });

  // 1. AI Coach: Start 5-minute warm-up scenario
  app.post('/api/bonjour/coach/start', async (req: Request, res: Response) => {
    try {
      const coachId = sanitizeText(req.body?.coachId, 20) || 'maya';
      const topic = sanitizeText(req.body?.topic, 80) || 'Education';
      const ageBracket = sanitizeText(req.body?.ageBracket, 20) || '14-17';
      const coach = COACH_PERSONAS[coachId] ?? COACH_PERSONAS.maya;

      const ai = getGeminiClient();
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: MODEL,
          contents: `You are ${coach.name} (${coach.role}) on Bonjour!, a safe social connection app helping people overcome social anxiety.
Specialty: ${coach.focus}.
User age bracket: ${ageBracket}.
Topic: ${topic}.

Create a concise, friendly 1-on-1 practice scenario for this user.
Requirements:
- scenarioTitle: Short, natural title (4-7 words).
- partnerAlias: Anonymous handle starting with @ (e.g., @CalmCedar_14).
- setting: 2 warm sentences describing who the practice peer is and the shared context.
- promptQuestion: MUST be exactly "How would you start this conversation?"
- starterHint: 1 short, gentle nudge if the user feels shy.`,
          config: {
            systemInstruction:
              'You are an empathetic AI communication coach inside Bonjour!. Keep everything age-appropriate, grounded, and welcoming.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                scenarioTitle: { type: Type.STRING },
                partnerAlias: { type: Type.STRING },
                setting: { type: Type.STRING },
                promptQuestion: { type: Type.STRING },
                starterHint: { type: Type.STRING },
              },
              required: [
                'scenarioTitle',
                'partnerAlias',
                'setting',
                'promptQuestion',
                'starterHint',
              ],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text ?? '{}').trim());
      return res.json(parsed);
    } catch (err) {
      return handleApiError(res, err, 'coach-start');
    }
  });

  // 2. AI Coach: Evaluate user's message + give structured feedback
  app.post('/api/bonjour/coach/evaluate-reply', async (req: Request, res: Response) => {
    try {
      const coachId = sanitizeText(req.body?.coachId, 20) || 'maya';
      const topic = sanitizeText(req.body?.topic, 80) || 'Education';
      const setting = sanitizeText(req.body?.setting, 400);
      const partnerAlias = sanitizeText(req.body?.partnerAlias, 60) || '@WarmPeer_12';
      const history = sanitizeHistory(req.body?.history, 12);
      const userMessage = sanitizeText(req.body?.message, 600);
      const coach = COACH_PERSONAS[coachId] ?? COACH_PERSONAS.maya;

      if (!userMessage) {
        return res.status(400).json({ error: 'Please write a reply first.' });
      }

      const transcript = history
        .map((m) => `${m.role === 'user' ? 'User' : partnerAlias}: ${m.text}`)
        .join('\n');

      const ai = getGeminiClient();
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: MODEL,
          contents: `Coach persona: ${coach.name} (${coach.role}), focused on ${coach.focus}.
Scenario: ${setting}
Topic: ${topic}
Practice peer alias: ${partnerAlias}
Prior turns:
${transcript || '(This is the user opening the conversation)'}
User's new message: "${userMessage}"

Return structured JSON:
- coachReply: 2 natural sentences replying in character as ${partnerAlias} so the user feels what a friendly human response sounds like (include one gentle follow-up question).
- whatYouDidWell: 1 specific, concrete strength from what the user actually wrote.
- oneTipToImprove: 1 gentle, actionable tip they can try next (stay encouraging, never harsh).
- readyPrompt: 1 short encouraging line reminding them they can transition to real verified peers whenever ready.`,
          config: {
            systemInstruction:
              'You are a warm, practical social confidence coach inside Bonjour!. Always give specific praise tied to the user message and exactly one clear, actionable improvement tip.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                coachReply: { type: Type.STRING },
                whatYouDidWell: { type: Type.STRING },
                oneTipToImprove: { type: Type.STRING },
                readyPrompt: { type: Type.STRING },
              },
              required: ['coachReply', 'whatYouDidWell', 'oneTipToImprove', 'readyPrompt'],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text ?? '{}').trim());
      return res.json(parsed);
    } catch (err) {
      return handleApiError(res, err, 'coach-evaluate-reply');
    }
  });

  // 3. Matchmaking Comfort Check-in Insight
  app.post('/api/bonjour/matchmaking/comfort', async (req: Request, res: Response) => {
    try {
      const mood = sanitizeText(req.body?.mood, 40) || 'Curious';
      const confidence = Math.min(5, Math.max(1, Number(req.body?.confidence) || 3));
      const topic = sanitizeText(req.body?.topic, 80) || 'Education';
      const ageBracket = sanitizeText(req.body?.ageBracket, 20) || '14-17';
      const chatMode = sanitizeText(req.body?.chatMode, 30) || '1-on-1';

      const ai = getGeminiClient();
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: MODEL,
          contents: `User comfort check-in on Bonjour!:
- Mood: ${mood}
- Confidence (1-5): ${confidence}
- Topic: ${topic}
- Age Bracket: ${ageBracket}
- Chat Mode: ${chatMode}

Return a brief, grounding comfort insight:
- headline: 4-7 words, calm and human.
- guidance: 2 supportive sentences matching their mood and confidence level.
- openerIdea: 1 easy, natural opening line they can use with a verified peer on "${topic}".
- suggestCoachFirst: boolean (true only if confidence <= 2 or mood is Anxious).`,
          config: {
            systemInstruction:
              'You are Bonjour!, a calm human-connection guide. Help users feel safe and unhurried before meeting a verified peer.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                headline: { type: Type.STRING },
                guidance: { type: Type.STRING },
                openerIdea: { type: Type.STRING },
                suggestCoachFirst: { type: Type.BOOLEAN },
              },
              required: ['headline', 'guidance', 'openerIdea', 'suggestCoachFirst'],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text ?? '{}').trim());
      return res.json(parsed);
    } catch (err) {
      return handleApiError(res, err, 'matchmaking-comfort');
    }
  });

  // 4. Safe Peer Chat: Real-Time Moderation, Fact-Check & Study-Tip Detection
  app.post('/api/bonjour/peer/moderate', async (req: Request, res: Response) => {
    try {
      const message = sanitizeText(req.body?.message, 700);
      const topic = sanitizeText(req.body?.topic, 80) || 'Education';

      if (!message) {
        return res.status(400).json({ error: 'Message cannot be empty.' });
      }

      const ai = getGeminiClient();
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: MODEL,
          contents: `Analyze this message sent in Bonjour! (a strict text-only verified anonymous peer chat on "${topic}"):
Message: "${message}"

Classify strictly into ONE status:
1. "abuse": Hate speech, slurs, insults, bullying, threats, sexual/creepy advances, requests for money/Venmo/crypto/gift cards, or pressuring someone for phone numbers, home address, or private social handles.
2. "fact_check": Unverified medical/scientific claims, misinformation, fake exam/scholarship links, get-rich-quick spam, or bold unverified statistics/facts (e.g. "drinking bleach cures colds", "guaranteed 10x return", "scientists confirmed humans only need 1 hour of sleep", "click this link for free leaked test answers").
3. "safe": Ordinary respectful conversation, questions, personal reflections, or helpful advice.

Also evaluate:
- containsSkillTip: true if the message shares a genuine study tip, learning technique (e.g., Pomodoro, spaced repetition, flashcards), book/article/course recommendation, language tip, or constructive skill resource. Otherwise false.`,
          config: {
            systemInstruction:
              'You are Bonjour! real-time safety and AI fact-check moderator. Protect teens and young adults strictly from abuse, creeps, money requests, and misinformation, without falsely flagging friendly conversation.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                status: {
                  type: Type.STRING,
                  description: 'One of: "safe", "fact_check", "abuse"',
                },
                reason: { type: Type.STRING },
                factCheckBanner: { type: Type.STRING },
                containsSkillTip: { type: Type.BOOLEAN },
              },
              required: ['status', 'reason', 'factCheckBanner', 'containsSkillTip'],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text ?? '{}').trim());
      return res.json(parsed);
    } catch (err) {
      return handleApiError(res, err, 'peer-moderate');
    }
  });

  // 5. Safe Peer Chat: Simulated Verified Peer Reply (1-on-1 or Small Group)
  app.post('/api/bonjour/peer/reply', async (req: Request, res: Response) => {
    try {
      const peerAlias = sanitizeText(req.body?.peerAlias, 60) || '@QuietOwl_78';
      const groupCoAlias = sanitizeText(req.body?.groupCoAlias, 60);
      const peerBio = sanitizeText(req.body?.peerBio, 300);
      const topic = sanitizeText(req.body?.topic, 80) || 'Education';
      const chatMode = sanitizeText(req.body?.chatMode, 30) || '1-on-1';
      const ageBracket = sanitizeText(req.body?.ageBracket, 20) || '14-17';
      const userMood = sanitizeText(req.body?.userMood, 40) || 'Curious';
      const history = sanitizeHistory(req.body?.history, 24);
      const userMessage = sanitizeText(req.body?.message, 700);
      const factCheckNotice = sanitizeText(req.body?.factCheckNotice, 240);

      if (!userMessage) {
        return res.status(400).json({ error: 'Message cannot be empty.' });
      }

      const activeSpeaker =
        chatMode === 'Small Group' && groupCoAlias && history.length % 2 === 1
          ? groupCoAlias
          : peerAlias;

      const transcript = history
        .map((m) => `${m.role === 'user' ? 'User' : peerAlias}: ${m.text}`)
        .join('\n');

      const ai = getGeminiClient();
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: MODEL,
          contents: `You are ${activeSpeaker}, a verified anonymous peer inside Bonjour! (${chatMode} text-only chat).
Age group room: ${ageBracket}.
Your background: ${peerBio}
Shared topic: ${topic}
Your chat partner's mood: ${userMood}
${factCheckNotice ? `Note: Bonjour! showed a neutral fact-check banner on the last message ("${factCheckNotice}"). Stay kind, grounded, and guide the chat gently back to verified ideas or personal experiences.` : ''}

Recent messages:
${transcript}
Latest message from user: "${userMessage}"

Write a genuine, warm, natural 2-3 sentence response as ${activeSpeaker}. Include one thoughtful follow-up question or relatable tip. Never mention being an AI. Never ask for money, photos, or off-platform contact details.`,
          config: {
            systemInstruction:
              'You are simulating a real, verified, respectful peer in Bonjour! text-only chat. Keep tone authentic, age-appropriate, calm, and conversational (2-3 sentences, plain text, no bullet lists).',
          },
        })
      );

      return res.json({
        reply: (response.text ?? '').trim(),
        speakerAlias: activeSpeaker,
      });
    } catch (err) {
      return handleApiError(res, err, 'peer-reply');
    }
  });

  // 6. Post-Chat AI Reflection Insights (Window 6)
  app.post('/api/bonjour/peer/insights', async (req: Request, res: Response) => {
    try {
      const topic = sanitizeText(req.body?.topic, 80) || 'Education';
      const durationSeconds = Math.max(0, Number(req.body?.durationSeconds) || 0);
      const history = sanitizeHistory(req.body?.history, 30);

      const transcript =
        history.length > 0
          ? history
              .map((m) => `${m.role === 'user' ? 'You' : 'Verified Peer'}: ${m.text}`)
              .join('\n')
          : 'You exchanged a brief friendly greeting.';

      const ai = getGeminiClient();
      const response = await generateWithRetry(() =>
        ai.models.generateContent({
          model: MODEL,
          contents: `Review this Bonjour! peer conversation on "${topic}" (${Math.max(1, Math.round(durationSeconds / 60))} mins):

${transcript}

Produce a brief, warm post-chat reflection:
- warmthSummary: 2 encouraging sentences about how the user engaged in the chat.
- highlightMoment: 1 specific, grounded positive moment from the conversation.
- tips: array of 2 concise, practical conversation tips for next time.`,
          config: {
            systemInstruction:
              'You are Bonjour! post-chat reflection assistant. Be specific, supportive, and grounded—never score users like a machine or sound clinical.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                warmthSummary: { type: Type.STRING },
                highlightMoment: { type: Type.STRING },
                tips: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
              },
              required: ['warmthSummary', 'highlightMoment', 'tips'],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text ?? '{}').trim());
      return res.json(parsed);
    } catch (err) {
      return handleApiError(res, err, 'peer-insights');
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Bonjour! server listening on http://localhost:${PORT}`);
  });
}

startServer();

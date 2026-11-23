-- Student rating (thumbs up/down) of an AI answer, stored on the answer itself (WBS 5.3).
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS rating VARCHAR(8);

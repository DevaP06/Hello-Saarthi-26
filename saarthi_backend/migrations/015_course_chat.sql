ALTER TABLE saarthi_conversations ADD COLUMN IF NOT EXISTS course_id INTEGER REFERENCES saarthi_courses(id) ON DELETE SET NULL;
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS course_id INTEGER REFERENCES saarthi_courses(id) ON DELETE SET NULL;
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS topic VARCHAR(128);
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS shared_query TEXT;
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS reply_to_id INTEGER REFERENCES saarthi_chat_messages(id) ON DELETE SET NULL;
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS answer_metadata JSON;
ALTER TABLE saarthi_chat_messages ADD COLUMN IF NOT EXISTS instructor_note TEXT;
CREATE INDEX IF NOT EXISTS ix_chat_messages_course_time ON saarthi_chat_messages (course_id, created_at);
CREATE INDEX IF NOT EXISTS ix_chat_messages_reply_to ON saarthi_chat_messages (reply_to_id);

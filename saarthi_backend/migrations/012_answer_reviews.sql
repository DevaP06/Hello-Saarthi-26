-- Instructor review queue for AI answers (model: AnswerReview).
CREATE TABLE IF NOT EXISTS saarthi_answer_reviews (
    id SERIAL PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES saarthi_courses(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES saarthi_users(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    reply_mode VARCHAR(32),
    confidence VARCHAR(16),
    reasons TEXT,
    source VARCHAR(24) NOT NULL DEFAULT 'low_confidence',
    status VARCHAR(16) NOT NULL DEFAULT 'open',
    occurrences INTEGER NOT NULL DEFAULT 1,
    correction TEXT,
    corrected_by INTEGER REFERENCES saarthi_users(id) ON DELETE SET NULL,
    corrected_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_saarthi_answer_reviews_course_id ON saarthi_answer_reviews (course_id);
CREATE INDEX IF NOT EXISTS ix_saarthi_answer_reviews_user_id ON saarthi_answer_reviews (user_id);
CREATE INDEX IF NOT EXISTS ix_saarthi_answer_reviews_status ON saarthi_answer_reviews (status);

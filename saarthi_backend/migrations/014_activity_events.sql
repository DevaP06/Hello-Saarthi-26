CREATE TABLE IF NOT EXISTS saarthi_activity_events (
    user_id INTEGER NOT NULL REFERENCES saarthi_users(id) ON DELETE CASCADE,
    event_id VARCHAR(36) NOT NULL,
    course_id INTEGER REFERENCES saarthi_courses(id) ON DELETE CASCADE,
    material_id INTEGER REFERENCES saarthi_materials(id) ON DELETE CASCADE,
    video_id INTEGER REFERENCES saarthi_videos(id) ON DELETE CASCADE,
    active_seconds INTEGER NOT NULL DEFAULT 0 CHECK (active_seconds BETWEEN 0 AND 30),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, event_id),
    CONSTRAINT activity_one_resource CHECK (material_id IS NULL OR video_id IS NULL)
);

CREATE INDEX IF NOT EXISTS ix_activity_events_user_time ON saarthi_activity_events (user_id, created_at);
CREATE INDEX IF NOT EXISTS ix_activity_events_course_time ON saarthi_activity_events (course_id, created_at);
CREATE INDEX IF NOT EXISTS ix_activity_events_material ON saarthi_activity_events (material_id);
CREATE INDEX IF NOT EXISTS ix_activity_events_video ON saarthi_activity_events (video_id);

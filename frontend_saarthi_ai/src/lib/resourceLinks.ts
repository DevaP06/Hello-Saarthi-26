function integerAtLeast(value: number | string | null | undefined, min: number): number | null {
    if (value == null || value === '') return null;
    const number = Number(value);
    return Number.isSafeInteger(number) && number >= min ? number : null;
}

export function resourceLink(source: {
    courseId?: number | string | null;
    materialId?: number | string | null;
    videoId?: number | string | null;
    page?: number | string | null;
    startSeconds?: number | string | null;
}): string | null {
    const video = integerAtLeast(source.videoId, 1);
    if (video) {
        const timestamp = integerAtLeast(source.startSeconds, 0);
        return `/videos/${video}${timestamp === null ? '' : `?t=${timestamp}`}`;
    }

    const course = integerAtLeast(source.courseId, 1);
    const material = integerAtLeast(source.materialId, 1);
    if (!course || !material) return null;
    const page = integerAtLeast(source.page, 1);
    return `/courses/${course}?tab=classwork&material=${material}${page ? `&page=${page}` : ''}`;
}

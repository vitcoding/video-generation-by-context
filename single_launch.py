from b_roll.b_roll_generation.broll_image_generation import (
    main as broll_image_generation_main,
)
from b_roll.b_roll_generation.broll_prompts import main as broll_prompts_main
from b_roll.b_roll_generation.kling_image_to_video import (
    main as kling_image_to_video_main,
)
from b_roll.b_roll_generation.video_editing_cloudinary import (
    main as video_editing_cloudinary_main,
)
from b_roll.b_roll_generation.word_level_transcriber import (
    main as word_level_transcriber_main,
)

if __name__ == "__main__":
    # word_level_transcriber_main()
    # broll_prompts_main()
    broll_image_generation_main(segment_ids=[2])
    # kling_image_to_video_main()
    # video_editing_cloudinary_main()

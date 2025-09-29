from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional

load_dotenv()

class Review(TypedDict):
    heading: Annotated[str,"Give the main heading of the para in a list"]
    key_themes: Annotated[list[str],"Give the key themes of the para"]
    summary: Annotated[str,"Give the precise review of the para"]
    sentiment: str

    Pros: Annotated[Optional[list[str]],"Give the Pros of the para in a list "]
    Cons: Annotated[Optional[list[str]],"Give the Cons  of the para in a list"]

    Decision: Annotated[str,"Whether a user/customer should buy it or not"]
    

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro') 
struct_model = model.with_structured_output(Review)

result = struct_model.invoke("""iPhone 17 Pro Max Launch: A Realistic Look at the Pros and Cons
Worst Paragraph Commentary:

The iPhone 17 Pro Max is an infuriating paradox: the best new iPhone, and yet somehow, the worst. You’re paying an exorbitant premium for a device that, at its core, feels like a frantic catch-up rather than a generational leap. The much-hyped "thermo-forged" aluminum body is not just a downgrade from the Pro's old steel and titanium, it's a "scratchgate" waiting to happen, with the new Cosmic Orange finish visibly scuffing after a week of normal use. Despite the vapor chamber and A19 Pro chip, sustained performance is barely a step up from last year—a pathetic showing when Android competitors are hitting new milestones. The supposed game-changing 8x telephoto is, in practice, a minor improvement over the iPhone 16's, with image processing that still feels overly aggressive. And, as always, Apple forces you to buy yet another proprietary fast-charging brick to take advantage of its "new" charging speeds. The whole experience feels like you're paying a record price for a phone that's a brilliant but flawed science experiment, all while a competitor like the iPhone Air offers a more innovative and elegant design for less.

Pros and Cons Analysis
Pros
Display: The iPhone 17 Pro Max features a stunning 6.9-inch LTPO OLED display with a peak brightness of 3,000 nits, making it incredibly visible even in direct sunlight. The inclusion of an anti-reflective coating is a welcome addition that further improves outdoor viewing.

Camera System: The new 48MP periscope telephoto lens with 8x optical-quality zoom is a significant step forward, offering more versatility and detail in long-range shots. The overall camera system, which includes a 48MP main sensor, is celebrated for its class-leading video quality and reliably great photo performance, especially with new on-device AI features.

Performance: The new A19 Pro chip, combined with an innovative vapor chamber cooling system and an aluminum unibody, delivers powerful sustained performance for gaming and demanding tasks. The inclusion of new Neural Accelerators in the GPU also makes it more powerful for on-device AI tasks.

Battery and Charging: The iPhone 17 Pro Max offers the best battery life ever in an iPhone, with up to 37 hours of video playback. Fast charging is finally more competitive, with the ability to reach 50% charge in 20 minutes with the new 40W adapter.

New Features: The "Center Stage" front camera now offers a multi-aspect sensor for more flexible framing, and the iPhone 17 Pro Max also comes with the new USB-C 3.2 Gen 2 port for up to 20x faster transfers.

Cons
Design and Durability: The new thermo-forged aluminum unibody, while beneficial for thermals, is a step down in perceived premium feel from the previous titanium frame. The new Cosmic Orange finish is also reportedly more susceptible to scratches, leading to a "scratchgate" issue for many early adopters.

Incremental Upgrades: For many, the phone feels like an incremental update. The camera improvements, while technically impressive, may not feel like a "game-changer" in everyday use, and the display, while brighter, is not a radical change from the previous generation.

Pricing and Accessories: The iPhone 17 Pro Max is an incredibly expensive device, and Apple continues its practice of not including a fast-charging adapter in the box, forcing consumers to purchase a separate, and often proprietary, high-wattage charger to take full advantage of the new charging speeds.

Lack of Innovation: While the phone has some new features, critics argue it lacks the "wow factor" and feels behind some of its competitors, such as the thinner iPhone Air or some of the more innovative Android foldables.""")

print(result)
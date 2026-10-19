About this project: Ups and Downs with Experience, Experiment, Learning and Mistakes 

If you are not interested in my experince of building this project, you may skip it.

I decided to read research papers to understand how the real world works and how researchers are making real progress. I stumbled upon (claude suggest me :) this paper called Visual Instruction Tunig Lui et.el 2023. I researched about it and found that if I want to understand this paper, I have to read another previous paper published by openAI called Contrastive Language Image Pre training (CLIP). I started reading it about 3 months ago. It was a rollar coster ride. As when I finally understands something it links to something I have first heard of. It was motivating and demotivating at the same time. It was fun.

It took me about two weeks to get through the CLIP paper. But now I finally understand how image encoding and hyperspace mapping works. 
After that I took a short break for about a week before starting Visual Instruction Tuning (Hero of the story). Though the paper was highly regarded, it was a easy read. I understood the logic pretty well and the intuition was spot on. After that I tried to study thier github repo and codebase. As I mention "I tried". It was all like a tangled earphone in some guys pocket. I decided to built each thing block by block using thier paper only.

The plan:
The paper's recipe has two stages: first align a vision encoder to a language model by training only a small connector on image captions, then teach the model to follow visual instructions. Spoiler(I kept that structure and shrank everything else to fit one GPU)

But the main villian of the story was the hardware that I didn't had. I choose what any other undergard unemployed student will choose - Google Colab. Also I had to cut short the datasets so it will fit in my Colab RAM. I tried to start building the code base, it was a learning curve, I have to read, learn and implements things (such as transfomers, data preprocssing etc) on the fly. It was not long, before I figured that my complete 1B model will not fit on 16 GB of VRAM along with other weights. So I decided to compromise. I learned aobout LoRA and QLoRA and how we can use small matrices to fine tune big LLMs without loading whole LLM parameters into GPU. It worked, my small A and B matrices were working as inteded. It took me about 1 month just to write the stage one code base (LLaVAMustBeCold). It was untested tough. There were few bugs and version mismatch, dtype mismatch, device mismatch, Cuda Out Of Memory troubles etc. I use CLAUDE/GEMINI to debug thoses and learned how these bugs and errors appeared in the first place and how to fix them. Also learned how to use gradient checkpointing to trade iteration speed with VRAM. I was a blast. Sometime I strech hair, beat the desk, and say few unhealthy words here and there. But in the end I got it to start the training loop for the projector. It was training, the progress bar was counting and loss is showing up.
It was such a wonderfull feeling, I can't describe in words.

Reaching Stage 1 was progress, but I was not quite there yet. I have to train Stage 2 (LLaVAMustBeHot) on even bigger dataset. I tried to do the same process: learn -> shrink -> build -> debug -> learn again.

It was solemn, that the available VRam on Google Colab was just not enough. I tried to use my college systems, but they had the very same GPU that I got on Google Colab for FREE. I was thiniking for renting a GPU, but before that I happended to talked to one of my senior PhD student. She told me that you should use kaggle as that provides free T4 dual gpu. I as amazed, I didn't knew that they do that.
I spend a whole week to shift my Colab code on Kaggle environment. Path system in kaggle was really a workout. I spend hours with claude and gemini to learn how to shift the code base from single gpu base to multi-gpu base, so it runs on dual GPU. I learned how to use dual GPU using ```accelerate``` and how to start two independent python process, how to work with different worker processes, how to enable communication between the GPUs so they work in sync and how to use mixed precision training. It was fun seeing thoses red bars going up and down in kaggle sessions.

I was training Stage 2 of the project. Progress bar was moving, losses were showing. I was delighted. I left my laptop open for about 15 hours, so that model can train unintruppted. That night I wasn't able to sleep. I was way too excited. The next morning, I saw the training was complete, with a big a** congratulation mesaage "YOU HAVE DONE IT!". I was soo excited to test the model, I frantically went to claude, and asked it to generate me the inference code snippet. I was in a hurry. But I just wanted to see my model works after 2.5 months of work.

IT WAS A FAILURE. After some debugs and test checks, I found out my model weights were all NaNs because old Tesla T4 GPU don't support bf16 precision. I was devasted. I even skipped college in hopes I can make it work some how. But it was a lost cause. I LEARNED IT THE HARD WAY, YOU SHOULD ALWAYS USE CHECKS BEFORE THE MAIN TRAINING RUN.

After draining my brain for over a day in hopes of fixing it, I chooses to take a break and went to sleep. I didn't bothered to open the code or research paper the next day. I did not give up. I needed some space for a while.

After a break of 2 days, I came back, had been decided to rent a cloud GPU even it costs hundreds (it was a risk, I didn't 100% know if it was hardware or my logic). I abandoned that dual GPU pipeline and scrape the Stage 1 projector. And rebuilt it to run on a good Nvidia RTX Pro 50000 Blackwell GPU on cloud notebook. I prepared my code for both the stages before hand, so to save expensive GPU time on cloud. I use Claude to build me security checks so that I can be sure the model is learning as intended. I bought $10 credits on Run Pod. Rented the GPU and started training. It was working on the screen, but I wounldn't have known what is brewing behind the scenes.

I trained both of the LLaVAMustBeCold and LLaVAMustBeHot within two days to save GPU time.
By the end of third day, it was finished. This time I wasn't even excited. I was focused. I build my own Inference script from scratch (not depending on claude this time) downloaded few images from the internet. Build the chat environment, though in a notebook. 

And finally I loaded the image and tried it asking ```What is in this image?``` it answered ```The image features a beach scene with a palm tree with a boat in background. The palm tree is located near the water, and the boat is floating on the water```
I was so surprised and not to mention happy, I can't tell. I jumped from the sofa and did some victory moves and gang signs (I don't know why).It was an image of palm trees on a beach. Instantenously, I thought, what if this was a fluke? I got serious and tested the model with few other images. IT WAS WORKING.

I chose the 1B model for iteration speed. A 7B model would have fit in memory with gradient checkpointing, but by my estimate each experiment would have cost about five times more, and I made enough mistakes that this mattered. A bug found after 15 hours hurts much more than one found after 3.
I have finally build something from scratch that mirror how the real works goes on in the world. Even thought it hallucinates over long chats, because it is just 1B parameter model behind the scene.

I have made it exist, now I have to make it beautiful. I cleansed all the debug codes here and there, cleared terminal commands and documented all the training scripts well. I have just published it as a GitHUB repo ```LLaVAMustBe```

End Comparisons --->
Paper: LLM -> Vicuna 7B / 13B | Stage 2 training -> full LLM fine-tune | Hardware -> multi-GPU A100 servers
Mine: LLM -> Llama-3.2-1B-Instruct | Stage 2 training -> LoRA + projector | multi-GPU A100 servers -> one RTX Pro 5000 (Blackwell, 47 GiB)


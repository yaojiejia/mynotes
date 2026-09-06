---
title: "Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 1: Overview, Tokenization"
source: "https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV"
author:
  - "[[Stanford Online]]"
published:
created: 2026-09-05
description: "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."
tags:
  - "clippings"
---
![](https://www.youtube.com/watch?v=JuoVZkPBiKk)

## Transcript

**0:04** · Welcome, everyone to CS 336, Language Models from Scratch.

**0:11** · This is the teaching staff.

**0:12** · I'm Percy.

**0:13** · This is Tatsu, Marcel, Herman and Steven, and we're bringing you the third edition of 336.

**0:19** · So we'll just do a quick round of introductions.

**0:21** · So I'd like to say that I've been doing language models for 20 years, but most of that time was small language models.

**0:29** · Actually, this is still small in the grand scheme of things.

**0:32** · I think when Tatsu and I started teaching this class two years ago, we weren't really sure what we were going to expect, but we were very pleasantly surprised that so many people wanted to learn how to build language models from scratch, especially in these days when a coding agent could probably zero shot a language model.

**0:49** · I'm really glad to see all of you here to actually want to learn how they work.

**0:55** · Yeah, thanks.

**0:56** · I'm Tatsu, I'm one of the co-instructors.

**0:58** · I'll be talking to you once we get to architectures and scaling and all these other very fun things.

**1:04** · I'm really excited.

**1:04** · This is the most fun class I've taught in my time here.

**1:07** · I think every year, Percy makes fun of me because he says you have to redo all your lectures because you took on architectures, and everything changes every time.

**1:14** · But it's actually pretty fun for me to do it.

**1:16** · And it's the first time that I've had that experience.

**1:18** · So I'm looking forward to going through this experience again with you all.

**1:23** · Hello, I'm Marcel.

**1:24** · I've seen this class once before, and I'm returning because it was so much fun last time.

**1:29** · \[LAUGHTER\] It's a lot of work though.

**1:33** · In my research, I do architecture stuff, I do higher order gradients, and I do training.

**1:38** · Yeah, looking forward to working with you guys.

**1:40** · Hello, everyone, I'm Herman.

**1:42** · A year ago, I didn't really how LLMs worked, so I think for me, tokens were the things you collect in video games, and attention was the thing you had, like the attention economy.

**1:54** · Then after spending a lot of time on this course last year, I'm now doing LLM research, and I'm really excited to TA this year.

**2:03** · Hi, everyone.

**2:04** · I'm Stephen.

**2:05** · I am a first time CA for this course and I'm very excited about it.

**2:08** · I think it will be a lot of fun.

**2:10** · Broadly, in my research, I work on language models, theory and some data efficiency stuff, and I'm excited to meet you all.

**2:18** · All right, so let's go into things.

**2:23** · So this is the third time we're offering it.

**2:26** · Last year, we decided to put all our lectures on YouTube.

**2:29** · So some of you have seen it.

**2:31** · So what's new?

**2:33** · Well, what hasn't changed is 'from scratch' philosophy.

**2:36** · We still believe strongly that by building everything from the ground up, you really learn how everything works.

**2:44** · Of course, we don't actually build everything up from scratch because that wouldn't fit in the quarter.

**2:52** · So over the last two years, we've been refining our recipe and figuring out what are the things that you build up from scratch that are the most high value?

**3:00** · And then finally, as Tatsu alluded to, even in a year, a lot has changed.

**3:07** · I think this year we're going to spend maybe a bit more time on mixture of experts.

**3:12** · And of course, agents are very popular these days.

**3:16** · So getting a handle on long-context and what is needed for that are going to be important.

**3:23** · So why did we make this course?

**3:26** · And I think the problem two years ago was that researchers were becoming disconnected from the underlying technology.

**3:36** · It used to be the case may 10 years ago, all AI researchers would just implement and train their own models.

**3:43** · And then even eight years ago, people would download pre-trained models such as Bert and fine tune them.

**3:51** · And I think a lot of today, you can get by by simply prompting a model.

**3:58** · And of course, there's nothing wrong with prompting model.

**4:00** · I think you can do amazing things with it.

**4:03** · I think moving up the abstraction in general is a great thing, but abstractions are leaky, and sometimes, I'm sure all of you have prompted models, you run into situations where something is-- you wanted to do something, but it just can't do it and there's no recourse.

**4:21** · And I would argue that if you're really interested in fundamental research, by simply prompting a model, you're vastly constraining the set of options, the design space you're looking at.

**4:32** · And by looking at fundamental research, you really need to tear up the whole stack.

**4:38** · So I would argue that full understanding of how language models work is really necessary for fundamental research.

**4:45** · And the way that we're going to get an understanding is by building.

**4:49** · That's the philosophy of the class.

**4:51** · But there's one small problem, which is that industrialization of language models has happened.

**4:57** · Frontier models are really, really expensive.

**5:00** · Even, this is three years ago.

**5:02** · GPT-4 was supposedly costing $100 million to train, and now costs probably are on the order of $1 billion, although that's speculative.

**5:12** · And the number of GPUs that all the big labs are building is just immense.

**5:18** · Furthermore, there's no details on how any of these models are built. So even back in 2023, the GPT-4 paper explicitly says that due to a competitive landscape and safety implications, we're not going to share anything about how the models are being built.

**5:33** · So these frontier models are in some sense out of the reach for us.

**5:38** · Now, we could build small language models, and we will build language models.

**5:43** · But I think it's important to remember that these might not be representative of the actual frontier models.

**5:49** · And I'll give you two examples why this might be the case.

**5:54** · So here's one.

**5:56** · This is from actually quite a long time ago.

**5:59** · I think back in 2021.

**6:01** · If we're going to spend more time on flop counting and looking at where the compute is being spent.

**6:06** · But if you look at small scales, the fraction of flops spent in the MLP layers is around 44%.

**6:14** · And if you scale up to 175B, then it goes to 80%.

**6:20** · So what you optimize and what matters at large scale is going to be different from small scale.

**6:25** · So if you did a bunch of small scale stuff here on the attention, you might not experience the same benefits at large scale.

**6:33** · The second example is that we know of emergence of behavior with scale.

**6:41** · So small models, this is, again, for a while ago, but even back then, if you have a 0 shot or a few shot learning of various tasks, it basically, seemed like nothing was working, and only when you reach a critical scale do you suddenly see a lot of improvement.

**7:00** · So again, if you're working on small scale, you might not see certain types of phenomenon compared to if you were working at full scale.

**7:09** · OK, so that might be a little bit disheartening, but fear not, we're going to learn something in this class.

**7:16** · And the question is, what can we learn that actually transfers?

**7:20** · And I think it's important to break this down into three types of knowledge.

**7:24** · First, there's the mechanics of how things work, what a transformer is, how model parallelism works.

**7:31** · There's mindset, which is, how do you go about approaching building a language model?

**7:37** · We're going to talk about how you want to squeeze the most out of your hardware, taking scaling seriously.

**7:42** · And then finally, intuitions.

**7:44** · Which data modeling decisions are going to yield good performance?

**7:49** · So now, in this class, I think we can do a pretty good job of teaching the mechanics, which is how things work and the mindset.

**7:57** · And we're going to really emphasize that you profile and benchmark everything and try to optimize for efficiency.

**8:03** · These things do transfer to a larger scale.

**8:06** · Now, the intuitions about what modeling decisions and what data decisions do work might not necessarily transfer across scales.

**8:16** · For that, you actually have to go somewhere where you can do things at scale.

**8:21** · So on the note of intuitions, it is worth remarking that some design decisions are just not justifiable, and just purely come from experimentation.

**8:35** · Whereas mechanics, you can, by construction, see how the parallelism and how kernels are going to speed things up and so on.

**8:44** · But for intuitions about what modeling changes work, I think you just have to run experiments.

**8:49** · There's this famous Noam Shazeer paper that introduces the SwiGLU activation, which we're going to look about.

**8:55** · And in the conclusion section, he very honestly has this final sentence which says, we offer no explanation.

**9:03** · We attribute their success, of these architectures, all else, to divine benevolence.

**9:07** · So that, in some sense, is something that you just have to gain from experience.

**9:15** · Final note about the bitter lesson, which I think has been circulating and people talk about it.

**9:21** · I think there's a common misconception here, what it means.

**9:27** · I think the wrong interpretation is that scale is all that matters.

**9:30** · Algorithms don't matter.

**9:32** · But that's not correct.

**9:33** · The right interpretation is that algorithms that scale are what that-- or what matters.

**9:39** · And you can think about very simply that the accuracy of your model is basically your efficiency times the resources.

**9:48** · So efficiency is output over input, and resources is the input.

**9:53** · And efficiency is actually, in some sense, way more important at larger scale, right?

**9:59** · If you're doing a small scale experiment, if your run takes twice as long, maybe you just wait twice as long and then you come back later.

**10:08** · But if you're doing things at scale, that could be hundreds of millions of dollars.

**10:13** · And you definitely don't want to do that.

**10:15** · Even like a 5% improvement might be a big deal.

**10:19** · So in fact, efficiency is actually really, really critical.

**10:22** · And I hope to bake that into your mindset as one of the consequences of this class.

**10:28** · And empirically, if you look at-- there's this paper from OpenAI in 2020 that showed there's a 44x algorithmic efficiency on ImageNet between 2012 and 2019.

**10:40** · And it's surely the case that hardware did get a lot better, but that also comes with algorithmic improvements.

**10:49** · And of course, when you multiply them together, that's when you see a huge bump in efficiency and accuracy.

**10:59** · So the framing with all that said is, what is the best model one can build with a certain data and compute budget?

**11:09** · For pre-training, it's mostly, we're going to talk about compute budget, because we're going to assume that we have a lot more data than we have compute.

**11:17** · But if you're in a setting where you're data limited or you have stashed away, actually, tons of B200s, then you might be data bound.

**11:29** · So in other words, maximize efficiency.

**11:31** · And we're going to see this theme come up throughout the class.

**11:36** · OK.

**11:38** · So next, I want to spend a bit of time talking about language models and a bit of a history and just contextualization before we jump into the more technical details.

**11:49** · So language models have been around for a while.

**11:52** · Shannon, back in the '50s was using language models to measure the entropy of English.

**11:58** · And for a long time, N-gram models were used, actually, in machine translation and speech recognition systems.

**12:07** · And they weren't the whole system, but they were an important part of making sure that you generated fluent text.

**12:13** · I would say that the lineage of the Modern Language models comes from neural architectures.

**12:20** · And so there is a bunch of ideas, I think, which are important to this development.

**12:28** · So in the '90s, there was LSTMs.

**12:32** · Yoshua Bengio, actually, wrote the first neural language model in paperback in 2003.

**12:38** · This is actually not an LSTM.

**12:40** · This was just a feedforward network that looked at the small context.

**12:45** · And then there was a Seq2Seq modeling, which boldly said, we can actually compress a whole sentence into a vector.

**12:53** · The atom optimizer attention mechanism, which was developed for machine translation.

**12:58** · The transformer architecture, which built on top of that, which was also developed for machine translation.

**13:03** · And then scaling up to a mixture of experts model parallelism.

**13:08** · You see a lot of different architecture and also systems and optimizer ideas developed in the 2010s.

**13:18** · And then by the late 2010s, I think, things were starting to get really interesting.

**13:23** · So there was the ELMo and BERT.

**13:27** · These are language models that were trained on lots of text, and then you could fine tune them on some downstream tasks like question answering, and it would show a huge improvement on that.

**13:37** · So the model there was, take one of these models and then fine tune.

**13:42** · And then Google had a paper that was really, I think, foreshadowing this view of prompt in, response out.

**13:51** · So it was really, I think, OpenAI that really opened up the floodgates here by embracing scaling.

**13:59** · So they had GPT paper back in, I think, 2018 or so.

**14:03** · They scaled it up to GPT-2, and then they really figured out or embrace the idea of scaling laws, which we're going to talk about in a second, which enable them to train GPT-3, which was a much more massive, almost more than 10x, the largest model at the time.

**14:26** · And it could show emergent behavior like in-context learning.

**14:30** · At that point, Google was saying, OK, we need to do something too.

**14:36** · So they trained a massive model.

**14:38** · It turned out to be undertrained.

**14:40** · And it turned out that their DeepMind, which was not integrated with Google at the time, had figured out optimal-compute, optimal scaling laws.

**14:49** · So all of this was happening.

**14:52** · And after GPT-3 came out, I think for many folks, this was kind of a wake up call.

**14:58** · And at that time, there were a lot of early attempts to, let's try to replicate this.

**15:03** · So there was a grassroots organization called Eleuther that created some open data sets and models.

**15:08** · They weren't very large because they didn't have much compute.

**15:11** · Meta first LLM, you could tell that it's a replication because it was like 175 billion parameters, but it was not a very good model.

**15:23** · They ran into a lot of hardware issues.

**15:27** · And then there was another, Hugging Face BigScience project.

**15:30** · So these models were, I would say, not very strong.

**15:35** · Then in the last three years, I think, the open model ecosystem has changed quite a bit, with Meta kind of leading the way with the Llama series of models.

**15:43** · Llama, Llama 2, Llama 3.

**15:46** · Mistral got in the game.

**15:47** · And then a whole set of Chinese models.

**15:53** · I think I'm missing some.

**15:55** · Like, ByteDance has something, and I think Tencent probably has some other stuff.

**16:00** · So it's hard to keep track of everything, but everyone's heard of DeepSeek and Qwen.

**16:04** · So I think I got the main ones.

**16:07** · But I think what's interesting and exciting about these is now we have open weight models that are approaching closed models.

**16:14** · So depending on who you ask and how you benchmark, they might be a little bit behind or comparable.

**16:19** · But they are definitely very, very credible models that are being widely used in industry.

**16:25** · Now there's another line of work, which is going beyond just releasing open weight models.

**16:33** · AI2, NVIDIA and the Marin project, which I work on, we try to provide not just the weights, but the paper and the code and the data so that we can understand how these models are built in a more thorough way.

**16:51** · Why do I emphasize this open ecosystem so much?

**16:55** · Mostly because this course would not be possible, I think, without these models by the fact that there are still many papers that are being published about how these big MOEs and RL systems are working enables us to at least glimpse into how these frontier models are being built, and trying to triangulate the pieces.

**17:17** · Now, I think, of course, a lot of the details, even with these Qwen papers, I think they're missing a lot of details.

**17:23** · So can't reproduce them.

**17:25** · Notably, the data mixture is something that we don't, but I think it's much better than nothing.

**17:32** · OK.

**17:32** · So in the last decade, I think, the idea of what a language model has changed, right?

**17:40** · It used to be something that you fine-tuned, and then it was something you prompt.

**17:45** · And now in the ChatGPT era, it was something you talked to and that you can have a conversation with.

**17:52** · And now-- OK, I guess I don't have internet.

**17:55** · \[CHUCKLING\] That's fine.

**17:58** · And now, we're in the era of agents.

**18:00** · If you click on this link, it basically shows you a giant agent trace.

**18:04** · And I'm still-- it's mind boggling how strong some of these models are.

**18:10** · You give it like a page of text and it does some really complicated agentic coding task.

**18:16** · So what we demand of our language models today is probably beyond the imagination of any one back 10 years ago.

**18:25** · That said, I think the fundamentals, I think, haven't changed that much.

**18:30** · Largely, we still built on GPUs and kernels.

**18:33** · We still optimize using gradient or stochastic gradient-like approaches.

**18:38** · We still have the transformer in attention, And we'll talk a little bit more about architectures, but it hasn't changed that much.

**18:46** · I think the specs are different.

**18:48** · Now we demand greater context lengths, which means that inference efficiency matters even more.

**18:55** · So the good news for us is that we didn't have to completely change this class.

**18:59** · Only Tatsu's section on the latest Chinese architectures.

**19:02** · \[LAUGHTER\] But the fundamentals, I think, are here to stay, at least for now.

**19:10** · OK, maybe I'll pause there in case there's any questions or thoughts.

**19:25** · OK.

**19:26** · So let's go on.

**19:28** · So a brief interlude.

**19:30** · So what is this program?

**19:32** · So this is what I call a executable lecture.

**19:34** · If you think it looks like a Python program, it is because it is actually a Python program, but it has been rendered for your viewing pleasure.

**19:45** · So when I step through it as actually executing the lecture.

**19:50** · And it makes us possible to basically step through code, which will be, hopefully, interesting and important later.

**19:57** · And you can also see the hierarchical structure of this lecture.

**20:00** · For example, we're done with this function.

**20:02** · Now, we go back to main.

**20:03** · OK, so let's talk about the course logistics, and the syllabus, which I think is going to take a good chunk of time.

**20:14** · So all the information is online at this website or cs336.stanford.edu.

**20:21** · This is a 5-unit class.

**20:23** · I think this, probably, class has a certain reputation, so I don't need to belabor the point too much, but this is-- we have five assignments.

**20:32** · They are pretty intense.

**20:34** · Even the first assignment according to this one.

**20:38** · Review was actually equivalent to the 5 assignments.

**20:42** · First, CS 224n.

**20:44** · I've been told, also, this is exaggerated, but better to, I guess, be conservative in your estimates.

**20:54** · So why should you-- so why should you take this course OK, so first you have an obsessive need me to understand how things work.

**21:02** · That should be your primary objective.

**21:04** · I think just pure curiosity on how language models work.

**21:08** · And then in doing the class, you'll develop a much stronger research, engineering muscles and have the confidence to go into a new setting and be equipped to deal with whatever situation comes up.

**21:22** · So I think when I started at Stanford, I created this class called Statistical Learning Theory, which was, basically, teaching the theoretical side of machine learning.

**21:32** · And that was nice because it equipped people.

**21:34** · So when you read a paper, you can understand all the math.

**21:37** · And now, since the field has shifted a lot more towards this more systems and empirical side of things, this is the kind of analogous class, which gives people enough depth that they feel that everything else seems kind of easy.

**21:52** · So why should you not take this course?

**21:55** · This is important because there's reasons you shouldn't take this course.

**21:58** · First, you actually want to get some research done this quarter.

**22:02** · You should probably talk to your advisor.

**22:04** · They should know you're taking this course.

**22:05** · Otherwise, there'll be-- probably, there might be some surprises.

**22:10** · You're interested in learning the hottest new techniques in AI.

**22:14** · I think there are many other great courses, seminar courses, topics courses that are good for that.

**22:20** · We don't do many of the things.

**22:22** · We don't do multimodality.

**22:23** · We don't talk about agents in any depth.

**22:27** · So if you want to learn about that stuff, this is not the right course for that.

**22:32** · If you come in and say, I have an application domain, I want to get good results on it, probably this is not the right course, at least to start.

**22:41** · I always recommend just prompt a model, fine tune a model.

**22:46** · And then as a last resort, you pre-train your own model.

**22:50** · Because it is a pain and it's also expensive, but it's a lot of fun.

**22:56** · OK.

**22:56** · So if you are not taking the class, you can self-follow along at home.

**23:03** · So all the lecture materials will be posted on the website.

**23:06** · Oops!

**23:06** · And they are also recorded through CGOE.

**23:10** · So thank you, CGOE, for doing that.

**23:12** · And later, they will be published to YouTube.

**23:18** · Now, of course, following at home and watching the lectures is great, but you learn, really, by doing the assignments.

**23:24** · So you'll have to figure out how to motivate yourself to do that.

**23:29** · OK, so speaking of assignments, we have five assignments.

**23:33** · The philosophy assignment is, how do we do from scratch, but not just, say, build a language model, and that's the assignment?

**23:42** · So we don't provide scaffolding code, but we do provide a bunch of unit tests to make sure that whatever you're building is actually correct, so that you don't get this sparse reward setting where you submit a homework and it's either correct or not.

**24:00** · What I recommend is that you can-- the assignments are structured so that much of the assignment can actually be done locally on your laptop.

**24:09** · You can implement it and check for correctness.

**24:11** · And then we are providing a cluster so that you can do an actual training run to see what the accuracy is, or get a bunch of GPUs to actually benchmark the performance of some kernel.

**24:25** · And then for fun, we will have, some leaderboards for most of the assignments, at least.

**24:34** · And they will look something like, well, now that you've learned about this topic, I'll try to minimize the perplexities given some of budget.

**24:42** · And both Marcel and Herman were masters of leaderboards back in the day, which was last year, or the year before.

**24:51** · So if you want tips on that, I'm sure they would be happy-- actually, I don't know, if they would tell you their secrets, but you can try.

**25:00** · OK, so last year, we were thinking about, well, what can AI do?

**25:07** · It was like, OK, well, I mean, yes, everyone can use AI, but just try to do your best.

**25:16** · I think now, I think, coding agents have gotten so good that they can just all solve all the assignments, right?

**25:21** · But to state the obvious, obviously, you're not going to learn anything if you just feed in the PDF Assignment 1 into a Cloud Code.

**25:31** · At the same time, AI can be very useful for answering questions and tutoring.

**25:35** · So we have to find a way to leverage AI.

**25:41** · So what we've decided to do is, we provide you a AGENTS.md file, or equivalently, a prompt which asks the AI to be pedagogically minded.

**25:53** · You can read more about it in our AI policy guide.

**25:56** · And the requirement is that if you're going to use AI, use it with this prompt.

**26:02** · And so it will answer questions about code, it will clarify any understanding, but it won't accidentally generate the transformer for you when the homework is to implement the transformer.

**26:14** · And this is the first year we're trying to do this, so please try it out and give us feedback if it's working or not working.

**26:23** · So compute.

**26:25** · So this year we have-- thanks to Modal, they have provided us with compute credits on their platform.

**26:33** · It's actually quite nice.

**26:35** · You can get a number of-- unlike last year-- well, I guess, you guys didn't take the class last year.

**26:41** · Last year, we had a cluster you SSH into.

**26:43** · This is using more of an API to-- which I was initially skeptical of, but looking at it, actually, is pretty pleasant to use.

**26:52** · So again, try it out and give us feedback on how it is.

**26:58** · We've written a guide on how to access and use the compute.

**27:04** · OK, any questions about course logistics?

**27:16** · OK.

**27:17** · All right, let's talk about what we're going to cover in this class.

**27:22** · So there's, basically, five parts mirroring the five assignments that you'll have.

**27:30** · Basic system, scaling, laws, data, and alignment.

**27:33** · So I'm going to now go through each part and just give you a taste of what you will learn.

**27:42** · So in the basics, this is basically the first two weeks.

**27:48** · The goal is to just be able to train a language model and build it from scratch.

**27:53** · So the components here are, we're going to tokenize the data, we're going to define architecture, and then we're going to implement optimizer and train it.

**28:00** · So then you wonder, what are the rest of the classes for?

**28:04** · Well, we'll get to there.

**28:06** · So let's start with tokenization.

**28:08** · The tokenization is really about, what are the atoms that the model operates on?

**28:14** · And formally, a tokenizer converts between raw inputs, which are just bytes, and a sequence of integers, which represent the tokens.

**28:26** · Conceptually, it's a segmentation of the text.

**28:30** · We're going to talk about the Byte-Pair Encoding, BPE tokenizer, which intuitively, breaks the input into frequently occurring chunks.

**28:41** · And then remember, this class is about maximizing efficiency, so through an efficiency lens, tokenization is good because it takes a long sequence, if you just think about the raw byte stream, and reduces it into a smaller number of tokens.

**28:55** · But more subtly, but maybe more importantly, it allows you to do adaptive computation.

**29:01** · So maybe some places are actually a lot of bites, but actually, it should be compressed into a one token, whereas some of the more rare or interesting parts of input should be left as multiple tokens.

**29:15** · OK, we'll talk more about this.

**29:17** · I just want to mention that every year, I'm hoping that I don't have to teach tokenization, because the dream is to really have an end-to-end way that directly operates on bytes.

**29:30** · And there's been a number of work, including recently, there's been these networks that seems promising, but so far, these have not been scaled to the frontier.

**29:42** · And since the frontier models are still using tokenizers, we felt like it would be still wise to teach tokenizers.

**29:50** · OK, so now after you tokenize your input, you have a bunch of tokens.

**29:56** · Now you define a model on top.

**29:58** · And everyone, I think, has a familiarity with the transformer.

**30:03** · And if you've taken 224n, the MLP class, then you've seen transformers.

**30:10** · Since then, I think, there have been a lot of improvements or refinements to transformers, which I think are important.

**30:18** · And Tatsu is going to talk more about this in a bit.

**30:20** · But just to run through a set of types of things that one might have to think about, so the activation functions have evolved.

**30:31** · How do you do positional encodings has evolved.

**30:35** · How you normalize has the different layers.

**30:40** · Blow up has evolved.

**30:43** · Instead of doing full attention, there's many ways to basically reduce the attention computation, because attention is n squared and where n is a sequence length, and that gets really expensive.

**30:56** · So there's a bunch of ideas around that.

**30:58** · If you're more ambitious, you can look at these state-space models or equivalently linear attention like Mamba and Gated DeltaNet.

**31:08** · These have become more popular in the last few years.

**31:12** · And usually, some hybrid between these models and attention seems to work quite well.

**31:17** · So, we'll be exploring some of that.

**31:20** · And then within the MLP layers of the transformer, the original transformer was just a dense MLP, and now a mixture of experts has become the dominant paradigm for building compute efficient transformers.

**31:34** · So we're going to talk about that.

**31:37** · And of course with a mixture of experts, it's not just defining architecture, but we'll see that we'll also need different techniques for training the model.

**31:48** · And then finally, perhaps somewhat boringly, but an important question is, what is the shape of your transformer?

**31:55** · How many layers?

**31:56** · How many heads?

**31:58** · What is hidden dimension?

**31:59** · Number of experts?

**32:00** · This might come in more as we talk about scaling laws, but setting these is actually-- seems kind of almost trivial.

**32:09** · It's a hyperparameter, but actually in the context of scaling and language models, it has a huge, huge implication.

**32:17** · So once you define your model architecture, how do you train the model?

**32:23** · And here there's a bunch of design decisions around the loss function.

**32:27** · There's next token prediction, which is the default.

**32:32** · But people have found that predicting more than one token seems to be helpful for improving the model.

**32:41** · And there's optimizers.

**32:44** · People used to use AdamW, but increasingly, Muon has been used, especially with some of the latest open models, such as the Kimi K2 models.

**32:57** · Initialization, which, again, sounds kind of boring, but turns out to have a huge impact on how your ability in the training stability of larger models.

**33:08** · Learning rate schedule.

**33:10** · Regularization.

**33:11** · Batch size.

**33:15** · And then MoE specific things.

**33:17** · So you look at this list and you might think, well, these are just hyperparameters.

**33:21** · I'm going to try a bunch of different options out.

**33:24** · But it turns out that really being very careful about setting these hyperparameters in a principled way will make the difference between a run that just blows up and is useless between a run that is achieving state of the art.

**33:42** · OK, I'll come back to that point when we talk about scaling loss.

**33:45** · So then in Assignment 1, what you're going to do is you're going to implement the BPE tokenizer, implement the transformer, the loss function, the optimizer, the whole training step.

**33:55** · We're going to make you do a bunch of resource accounting so you understand where your flops are going.

**34:01** · You're going to train some models on these data sets like TinyStories and OpenWebText.

**34:06** · And then there's going to be a leaderboard where you're going to try to drive down perplexity as fast as you can.

**34:15** · So for those of you familiar with NanoGPT, speedruns, it's kind of similar to that.

**34:24** · OK, so by the end of Assignment 1, you should be able to walk away and build a language model from scratch.

**34:32** · So that's very exciting.

**34:36** · If I have a high level takeaway here, it's that while the tokenizer and modeling and training are presented as distinct pieces, it is actually, everything is about balancing the following.

**34:53** · So you want expressive models because you want to represent the complexities of the data, but at the same time, you want your training to be stable.

**35:04** · And we're going to talk a lot about how do you keep the parameter and gradient norms in this goldilocks zone so they don't blow up and don't vanish.

**35:13** · Turns out, a lot of training language models is about just stability.

**35:20** · And then finally, efficiency, which is somewhat more straightforward.

**35:25** · You just make it run fast on hardware.

**35:27** · But you're going to see interesting things like, if we change the architecture, a lot of the architecture decisions are, well, we can make it faster by, let's say, reducing the projecting of a low dimensional space.

**35:41** · But then the question is, does it work as well?

**35:44** · And so making those trade offs is something that-- is the name of the game here.

**35:53** · So in Assignment 2, we're going to dive more deeply into systems.

**36:03** · And the goal here is just to get most out of your hardware.

**36:05** · So we're going to talk about kernels, how you parallelize across multiple GPUs, and how you do inference.

**36:12** · So the basics, which we're actually going to start on next lecture, I mentioned resource accounting, and I think you've all probably built models before.

**36:26** · But this is really about keeping track of where all the flops go and where all the memory is being spent.

**36:35** · So we're going to spend some time, basically, doing the resource accounting.

**36:39** · We're going to see this formula that comes up, how many flops does training SMDB on model on 1 trillion tokens?

**36:48** · Well, it's six times n times D roughly.

**36:51** · And where does that come from?

**36:53** · And then we're going to look at the hardware.

**36:58** · And here's a cartoon I picture of what to remark about hardware, is that, your memory is not where your compute is, and you have to move either your parameters or activations from the memory to compute, do the compute, and move it back.

**37:15** · And that often is the bottleneck.

**37:20** · So for example, B200, which we'll have the opportunity to play with, it has 2.25 petaFLOPS per second if bf16.

**37:30** · And has 8 terabytes second of memory.

**37:33** · So what does that mean?

**37:35** · I think when we-- I think I'll do this next lecture.

**37:39** · We're going to break this down and use this information to do some calculations and see how long different types of algorithms will need to take.

**37:49** · We're going to talk about roofline analysis, which allows us to understand whether a computation is bottlenecked by either a compute or memory.

**37:57** · In general, it is memory.

**38:02** · And then talk a little bit about benchmarking and profiling.

**38:08** · So here's what a DGX B200 looks like.

**38:13** · You have 8 GPUs.

**38:15** · They're connected via MV link.

**38:19** · And then if you have many-- if you have a thousand GPUs, then you would have multiple of these and they're connected either InfiniBand or ethernet.

**38:28** · So the next two parts when we talk about system is kernel.

**38:32** · So a kernel is basically a function that runs on the GPU.

**38:36** · And when you're just using plain PyTorch, all the PyTorch primitives actually correspond to launching particular kernels, which are built in.

**38:47** · So you're already using kernels, whether you it or not.

**38:54** · But the point is that, for certain types of computation, if you look at it, you can actually write custom kernels to make the GPUs go faster.

**39:05** · And the main principle here is organizing the compute to minimize data movement.

**39:11** · So remember, this picture moving data from memory is expensive.

**39:19** · So you want to try to minimize that.

**39:21** · So just as a simple example, suppose you wanted to compute A and B. So often, you would have to read from your high bandwidth memory, HBM, compute it, write it back, and then read again, compute it, write it back.

**39:36** · So then, you're basically sending the data back and forth twice.

**39:40** · And there's this idea called fusion, where you read it once, do both of the computations, and you write it back.

**39:47** · And that will save you a lot of time.

**39:52** · So that's operator fusion.

**39:53** · Tiling is a more sophisticated variant around the same idea.

**39:59** · There's also-- GPUs have also gotten a lot more complicated.

**40:03** · And I'm not sure how many of these details we'll have time to get into, but at least we want to expose you to some of the peculiarities, I would say, of GPUs and give you an appreciation for the types of things that one has to consider in order to squeeze as much juice out of them as possible.

**40:20** · And we'll write some kernels in and Triton.

**40:25** · So what happens if you have thousands of GPUs?

**40:31** · So the principle of minimized data movement is still the same.

**40:36** · The only thing is that moving data between different GPUs is even more expensive.

**40:44** · We're going to talk about how these very classic collective operations, gather and reduce and all-reduce are the way to think about, basically, distributed training.

**40:57** · The general game is that we have these model parameters, We have activations, gradients and optimizer states, and they need to be sharded or split across multiple GPUs.

**41:08** · And of course, you need to bring the right data to the right nodes to make the compute and write it back.

**41:13** · So there's a whole kind of orchestration, and how to do that efficiently is going to be the topic of this unit.

**41:23** · And there's multiple ways to shard.

**41:26** · You can shard by splitting up your data, splitting up your model, splitting up different layers in the model, splitting up the sequences, splitting up between experts.

**41:34** · And we'll talk about the trade offs that come with each of these.

**41:38** · OK.

**41:39** · And then finally, we're going to talk about inference.

**41:43** · Which, as I mentioned, is growing in importance.

**41:46** · So the goal of inference is to actually use the model.

**41:49** · So minor detail here.

**41:52** · So inference is, of course, you need to use inference when you're chatting with a model, but it also is useful for reinforcement learning.

**42:01** · It's useful for doing the rollouts.

**42:05** · Test-time compute, generating synthetic data, evaluation.

**42:07** · So inference is a very critical part of what it means to do language modeling work.

**42:16** · So we're probably not going to spend as much time as I like on this because the course is already kind of filled up, but we'll see what we can do.

**42:26** · There was some discussion around whether we should make you write inference from scratch, but we'll see.

**42:35** · So the way to think about inference is that there's two phases, a prefill and a decode.

**42:40** · In the prefill, You take the prompt and then you feed all the tokens forward and build key value pairs.

**42:46** · This is very much like what happens in training.

**42:49** · And then in the decoding part, tokens are generated one at a time.

**42:55** · And this is the part that becomes quickly memory bound, and this is why inference is hard.

**43:03** · And so there's many things you can do to speed up inference.

**43:08** · You can try to use a cheaper model by pruning a larger model or you can quantize, you can distill.

**43:14** · You can use this technique called speculative decoding, where you use a cheaper model to run ahead and guess a bunch of tokens.

**43:21** · And now, you use the full model, which can operate on those tokens in parallel to see if it's good.

**43:28** · And if you got lucky, then you can accept all those tokens and you're much faster than if you're doing one token at a time.

**43:35** · And of course, you can do systems optimizations.

**43:38** · There's a bunch of kernels that are designed specifically for inference.

**43:41** · And then one of the interesting things about inference is that, if you're running a service, queries are coming at you at potentially different times, and then you have to figure out how to batch them up.

**43:54** · Whereas in training, you basically already define your batches and everything is much more predictable.

**44:02** · So in Assignment 2, there's going to be implementing kernels in Triton and doing some of parallel training.

**44:14** · The details here might actually change.

**44:16** · As the CAAs have grand plans for revamping the systems.

**44:24** · So the assignment might look a bit different from last year's, but it will cover roughly the same material.

**44:33** · One thing I will mention is that there's this wonderful book out of some Google people called How to Scale Your Model, and I think it's really nice for providing a conceptual understanding of roofline analysis, and transformer math, and doing LLMs conceptually.

**44:51** · So I highly recommend you take a look at that.

**44:53** · Now, the only thing is that it's from Google, so it's about TPUs.

**44:58** · But a lot of the high level concepts are similar.

**45:03** · And now, they have a new chapter on how to think about GPUs.

**45:09** · OK.

**45:12** · So the third assignment is about scaling laws.

**45:15** · So by now, you've trained a language model.

**45:19** · You can make it go really fast by optimizing kernels in parallel.

**45:25** · Now, you want to scale up.

**45:28** · So how do you scale up?

**45:30** · So imagine the following setting.

**45:32** · If you had 1e25 FLOPs, so this is tens of millions of dollars of compute, what model would you train?

**45:42** · So this is, I think, a daunting task because if you mess up, well, that's a lot of money down the drain.

**45:49** · And you can't do your, probably, typical hyperparameter tuning at that scale because you only get to train one model.

**45:58** · And so this is the key problem that you have to deal with in large language model training that you don't have to really deal with if you're just fine tuning a model or doing small scale stuff.

**46:11** · And so the key conceptual shift here is that we shouldn't think about a single model that we're training, but really think about a scaling recipe.

**46:20** · And a scaling recipe is a mapping from a FLOP budget, let's say 1e25 or 1e24, to a set of hyperparameters.

**46:28** · Basically a config file.

**46:30** · And for a given scaling recipe, what we will do is run a bunch of experiments to compute the loss that you get at smaller scales, and then you fit a scaling law, and that enables you to predict the loss at a target scale.

**46:47** · So maybe you run some small experiments, you fit a scaling law, and then you predict out what you're going to get at a larger scale.

**46:56** · So that's the primitive.

**46:58** · Now using this, what you can do is now you can optimize the scaling recipe, targeting a larger scale using smaller scale experiments, which is wonderful.

**47:09** · And second of all, you can predict the loss that you're going to, in theory, achieve before actually running the experiment.

**47:19** · Which allows you to go raise money.

**47:23** · And you say, well, look, I ran the small scale experiments, and I think I can get really-- like, a GPD-5 level model.

**47:29** · Please give me a lot of money so I can train that model.

**47:35** · But one thing I think is maybe another misconception is that scaling laws are not laws of nature.

**47:43** · They don't just happen automatically.

**47:46** · You have to will them into existence.

**47:49** · And this happens by careful construction of a scaling recipe.

**47:54** · And the scaling recipe, remember, it has to extrapolate.

**47:57** · So what this typically means is that you have a sequence of hyperparameters which, say, as the scale increases, maybe the learning rate is a constant, maybe it drops, maybe the batch size increases by how much.

**48:09** · And these are things that a scaling recipe has to figure out.

**48:15** · And so you parameterize.

**48:18** · So then in order to get these predictable scaling laws, one thing that you actually have to think about is how do you parameterize the model in a way to get what's called hyperparameter transfer?

**48:31** · Meaning that the hyperparameters you use at small scale are either the ones that you use at larger scale or are predictable functions of it.

**48:38** · Because if every scale, your learning rate is sometimes 1e-5, and sometimes 1e-4, then you're not going to be able to magically guess the right learning rate when you are at larger scale.

**48:50** · So one shift in thinking is that predictability is actually at least as important as optimality.

**48:56** · So you normally think, oh, we're trying to optimize for efficiency here, and we want to hyperparameter tune and make things optimal.

**49:05** · And yes, you do want to do that, but you also want this predictability so that you don't get surprised at larger scale.

**49:13** · So this is state setting.

**49:17** · The actual scaling laws we're going to look at are fairly classic.

**49:22** · So these are-- some of you might have seen this idea of, well, if I give you a FLOPs budget, should you train-- how should you balance training a larger model versus training on more tokens?

**49:36** · And this is where the classic compute optimal scaling laws from Kaplan et Al and the so-called Chinchilla scaling laws comes in.

**49:43** · The basic idea is that you for each FLOPs budget, so let's say 6e18 all the way to 3e21, you sweep across different model sizes and you choose the best one.

**50:05** · So think about minimizing each of these.

**50:08** · And then you fit a curve that allows you to, basically, predict the number of parameters given a FLOPs budget.

**50:16** · And if you're lucky, it will rely roughly on the line.

**50:19** · And if you're unlucky, it's going to be all over the place, which means that you should have no confidence that you're going to be able to predict reliably.

**50:27** · And so the upshot of this, this is quite crude, but a rule of thumb is, the 20 times the number of parameters is the number of data points you should train on.

**50:45** · So a 70B parameter model should be trained on roughly 1.4 trillion tokens.

**50:50** · Of course, depending on the data set and architecture, this number will actually vary.

**50:56** · Also, this doesn't take you into account the inference cost.

**51:01** · A lot of models these days are small, but they're trained on way more tokens than is compute optimal because you want a smaller model for inference reasons.

**51:09** · So one fun thing that we've been doing in the Marin project is pre-registering our results.

**51:16** · So we fit a bunch of scaling plots at different compute budgets.

**51:22** · We fit a scaling law, and we basically made these predictions out to 1E22 FLOPs.

**51:30** · This one is actually training.

**51:31** · If you go to the Marin website, you can follow along.

**51:34** · It should actually be done maybe as early as tonight.

**51:37** · So maybe on Wednesday I'll report back on how we did and see how we measure-- how we match the pre-registered loss.

**51:47** · So the idea here is that we made a prediction on if we were to train this large model, which we've never trained before.

**51:53** · If we can predict how well it's going to do, then that's really nice.

**52:00** · So in Assignment 3, I think this is either fun or actually, it's a fun-- let's just say it's a fun assignment.

**52:09** · \[LAUGHTER\] So we're going to define this training API, which basically, you give us hyperparameters, and we're going to give you a loss back.

**52:20** · So what we're going to try to do is simulate what happens if you could do a lot of training runs.

**52:25** · And of course, we don't have enough compute to actually allow everyone to train their own API model or anything.

**52:33** · So basically, we trained a bunch of models offline, and we, basically, provide this cache.

**52:39** · So it looks like you're training.

**52:41** · And what you're going to do is submit training jobs.

**52:45** · You give us a config, basically, we give you a loss back and then you could do whatever you want.

**52:50** · We would recommend you fit scaling laws to these points.

**52:56** · And then you extrapolate.

**52:57** · And then we give you a budget, and we basically evaluate how well your model landed.

**53:04** · So it is meant to-- I was going to say, it's meant to replicate the high stress scenarios if you actually had a budget like $100 million that you need to spend.

**53:16** · You have to be very careful about how you spend this.

**53:21** · Of course, this is low stakes.

**53:28** · So at this point, you will have, you train a model, how to make it fast, you know to scale up.

**53:38** · Now, what's missing?

**53:39** · What do you train the model on?

**53:42** · And that's going to be the subject of the data section, which is arguably one of the most important things because data quality basically specifies how good your model is going to be.

**53:56** · One way to also frame it is, what do you want your model to do?

**54:02** · Data, basically, reflects what your model wants.

**54:07** · So do you want to speak multiple languages?

**54:10** · Be good at having a conversation?

**54:13** · Do you want to run long agentic coding tasks?

**54:17** · And so part of that is also going to-- so we're going to start by talking about evaluation, which, basically, defines the capabilities that you'd like your model to have.

**54:30** · One thing we'll talk about is, evaluation is a fairly deep topic.

**54:38** · It's not just about running on some benchmarks.

**54:42** · There are internal evaluation metrics for model development.

**54:46** · And what matters here is that smoothness across scales so that there-- remember, we want things to be predictable.

**54:55** · Relative performance matters.

**54:57** · You don't care, necessarily, how well this does in absolute terms, because, let's say, a perplexity number is just like, what is a perplexity of 1.2 on some held out data really mean?

**55:12** · And then there's external metrics.

**55:14** · These are things that you report to your customers or your reviewers or whoever you're presenting your thing to.

**55:23** · And here, ecological validity really matters.

**55:26** · And I think sometimes these two things get conflated, but I think they really serve two distinct purposes.

**55:31** · And so you can think about perplexity as something that is really helpful for internal development.

**55:38** · And still to this day, perplexity is a very good way of capturing the intrinsic quality of a model without getting-- worrying about benchmarking.

**55:51** · Now, there's a separate question of what you run your evals on.

**55:57** · And the recommended thing is, well, if you have some data that's not in the internet, that would be good because you can avoid contamination.

**56:05** · OK.

**56:06** · So and then there's advanced use cases which are more representative of external facing use cases.

**56:17** · So one thing to also note is that language models are purportedly very general purpose.

**56:23** · So it's only fitting that we actually need a very diverse set of evaluations.

**56:30** · And I would always recommend having many evaluations that look at-- you can average them into a single number, but often that average conflates a lot of different things.

**56:40** · So now after we set up the evals, we know what we're building.

**56:45** · How do you get the data?

**56:48** · Well, first thing is that data does not just fall from the sky.

**56:53** · It has to be actively curated.

**56:57** · Often, I think, especially in classes and also in research, sometimes you're just given a data set, and then it's like, OK, well, now, I do stuff on the data set.

**57:07** · But a lot of language models, especially if you want to collect these large data sets, you have to go and actively look at it.

**57:14** · So web pages are crawled from the internet.

**57:17** · There's books, I guess, which is controversial at this point, but arXiv papers, GitHub code, and so on.

**57:23** · This is an old figure from the pile from 2021.

**57:27** · And you can see, language modeling data sets are fairly diverse.

**57:33** · There's, especially these days, I think, a lot of contention around, is it fair use to train on copyrighted data?

**57:41** · Maybe sometimes you have to license data and so on.

**57:44** · So there's legal issues around data which, I think, are quite important.

**57:50** · For example, a lot of GitHub code doesn't have a license.

**57:53** · So how do you interpret that?

**57:55** · Do you assume it's permissive, or do you be conservative and assume it's not permissive?

**58:01** · So there's also-- the fact is that data is not even text.

**58:09** · It's either HTML or PDFs, or a code is directories.

**58:12** · And this requires processing to turn it into actual text to be usable for training.

**58:18** · So that's the topic of data processing.

**58:21** · There's a few steps that has to happen here.

**58:24** · Transformation, converting some nontexting in a text.

**58:30** · Filtering, keeping only the good stuff.

**58:34** · If the random internet document in Common Crawl is extremely bad and you don't want to train on it, most likely.

**58:44** · You want to deduplicate.

**58:46** · There's multiple sources, or how do you combine the different sources?

**58:51** · And then finally, more recently, there's been a lot of work on generating synthetic data, which could mean taking the real data and just rewriting it into things that are more the downstream task or just more Wikipedia like, or whatever you want.

**59:09** · So this is an active area of research.

**59:12** · Also, data can be used both for pretraining.

**59:16** · Something called mid training, which is usually the high quality data that you put at the end of the pretraining step.

**59:25** · And this includes long context data such as maybe BigCode repositories or books.

**59:31** · And then finally, there's post-training data, which are maybe conversations or agentic traces with tool calling.

**59:40** · So in Assignment 4, we're going to make you start with a very raw corpus, like a rock web crawl, and do all the work to filter and to dedupe, and make the data clean.

**1:00:00** · So this is, I would say-- I don't know if I would call it not fun, but it is certainly, I think, a lot of what people would call dirty work.

**1:00:13** · But that is, I think, an important part of building a language model from scratch so you have to get the full experience.

**1:00:20** · So finally, alignment.

**1:00:25** · So far we've, basically, trained a model using full supervision.

**1:00:30** · Predict the next token or the next few tokens.

**1:00:33** · Now at this point, the model should already be reasonable, but we can improve it further by using weak supervision.

**1:00:43** · And why weak supervision?

**1:00:44** · It's because sometimes it's easier to critique than it is to generate.

**1:00:49** · So you can't always have data that says, this is the right response to this prompt, but maybe you can have a way of specifying what good looks like.

**1:00:58** · So then, the basic template is that you generate responses from the model.

**1:01:02** · You score them either with a human or verify or a LM judge, and then you update the model to prefer better responses.

**1:01:09** · This can be instantiated either through various RL algorithms such as PPO or GRPO, or in a simpler, for preference data, DPO.

**1:01:22** · So the challenges around RL are that RL algorithms are unstable and hard to tune.

**1:01:32** · Some of you probably know this from firsthand experience.

**1:01:36** · Personally, I prefer to keep things as much in the fold so we supervise the case as long as possible.

**1:01:41** · And then finally, OK, fine, I have to do RL.

**1:01:44** · But some people, for whatever reason, like doing RL.

**1:01:49** · Also, what we'll, hopefully, talk about this year is that if you do RL at scale and try to maximize your throughput, there's actually a lot of systems challenges.

**1:02:01** · You have to have an inference server and a training server.

**1:02:04** · And then the inference server has to generate these rollouts.

**1:02:07** · Especially if you do RL against environments that involve code execution.

**1:02:11** · It's a whole kind of orchestration game.

**1:02:13** · And then if your workers lag behind or then you get into off policy issues, and then you're constantly kind of juggling this on-policyness with a desire to maximize throughput.

**1:02:28** · It's a big, wonderful mess, which, hopefully, we'll talk more about when we come to that.

**1:02:33** · So Assignment 5, we're still deciding what exactly we want to do.

**1:02:40** · Last year, it was implemented DPO and GRPO, and get it working for some math benchmark, but we'll see how much farther we can push it on the realistic dimension this year.

**1:02:54** · So again, remember it's about efficiency.

**1:02:58** · And efficiency can be either data efficiency or compute efficiency.

**1:03:03** · So the way to think about it, you have all these resources.

**1:03:06** · you have data, you have the hardware, which has compute cores, you have memory, communication bandwidth.

**1:03:12** · And you're just trying to figure out how do you build the best model according to some evaluation given a fixed set of resources.

**1:03:20** · So through this lens, I think, you can actually think about a lot of these design decisions as optimizing for this.

**1:03:29** · So systems, clearly, that's about compute efficiency.

**1:03:33** · Tokenization, as I mentioned before, you can't just work with raw bytes, but that's going to be very compute inefficient, at least with today's model architectures.

**1:03:46** · And so a lot of tokenization is about improving compute efficiency.

**1:03:50** · Model architecture, many of the changes that we'll see are motivated by reducing the memory or flops.

**1:03:58** · In fact, a lot of them are influenced by the need to have faster inference.

**1:04:04** · Data filtering, you can also view as-- through efficiency lens.

**1:04:07** · We don't want to waste time updating gradients on a bunch of redundant bad data, even if it might not hurt you, but it hurts you in the sense that if you have a fixed compute budget, more time on bad data means less time on good data.

**1:04:21** · And then finally, scaling laws is explicitly about how you can, essentially, do effective hyperparameter tuning on much smaller models.

**1:04:30** · And now tomorrow, we might become data-constrained, and the calculus of what design decisions you should take might change.

**1:04:37** · But I think the overall mindset which we're trying to teach you is think about the efficiency of your approach.

**1:04:46** · OK, let me stop there.

**1:04:49** · Are there any questions about any of these assignments or topics?

**1:05:05** · OK.

**1:05:06** · So now let's do tokenization.

**1:05:10** · So this is-- we're jumping into our first unit here.

**1:05:15** · So Andrej Karpathy has this really good video on tokenization.

**1:05:18** · You should check it out.

**1:05:21** · So starting point is raw text is what is text?

**1:05:25** · It's Unicode strings.

**1:05:28** · And on the other hand, the language model places a distribution over sequences of tokens usually represented as indices.

**1:05:35** · So we need a procedure that encodes these strings into tokens, and also a procedure that decodes tokens back into strings.

**1:05:44** · So a tokenizer is basically something that can do this round trip.

**1:05:50** · So here are some examples to give you a flavor for how tokenizers work.

**1:05:54** · Actually, I should have tried to get in there earlier.

**1:05:56** · So this is not going to work.

**1:05:58** · If you go to the site you can play around with different tokenizers.

**1:06:03** · So some observations here.

**1:06:07** · And you'll appreciate why tokenizers are kind of annoying and why people want to get rid of them.

**1:06:14** · So a word and its-- a word conglomerate with its preceding space are different tokens.

**1:06:24** · So many tokens you'll actually see are space a word, which is fine, but kind of strange.

**1:06:32** · So this hello and this hello is actually two completely different indices that have nothing to do with each other.

**1:06:40** · And sometimes, depending on the tokenizer you use, numbers are represented with-- every few digits is a token.

**1:06:50** · Sometimes, it's predictable, and sometimes, it's not.

**1:06:52** · Some tokenizers try to make every digit a token, but then you're blowing up the number of tokens you have.

**1:06:57** · So there's some trade off there.

**1:07:00** · So here's the GPT-5 tokenizer.

**1:07:04** · So you can take this string and convert it into these indices.

**1:07:14** · And then you can decode it back into the string.

**1:07:18** · And so tokenizers should round trip.

**1:07:20** · If you implement tokenizer, it doesn't round trip, you have a problem.

**1:07:25** · So the compression ratio here is the number of bytes per token.

**1:07:32** · So in this case, we have, the number of bytes of this string is 20.

**1:07:38** · The number of tokens is 8.

**1:07:40** · And you do 20 divided by 8.

**1:07:41** · So the compression ratio is 2.5.

**1:07:44** · OK, so 2.5 bytes per token.

**1:07:46** · The larger the compression ratio, that means the shorter the sentence, which is good because attention is quadratic, and you want to make sure the sent sequence is shorter.

**1:07:56** · Now, you could obviously increase the compression ratio by increasing the vocab size, but then, you get into sparsity where more and more-- because every element of vocab is treated like a distinct element.

**1:08:10** · So these days, tokenizers, especially multilingual tokenizers, have 100k or 200k tokens, distinct tokens.

**1:08:22** · So you can look at the GPD token vocabulary.

**1:08:25** · I think we'll skip it in the interest of time.

**1:08:28** · So how do you build a tokenizer?

**1:08:30** · So I'm going to go through this fast.

**1:08:34** · So the first thing you might do is like, well, Unicode string, that's a sequence of Unicode characters.

**1:08:40** · And each character is an integer, which you can call ord in Python, and you get some number out.

**1:08:47** · And this can be converted into characters.

**1:08:49** · So let's just build a character level tokenizer which basically breaks up each character and encodes it in a token.

**1:08:57** · And then this can decode back.

**1:08:59** · Life is good.

**1:09:01** · So now, there are 150k Unicode characters.

**1:09:05** · So your vocab size could be 150k, which is a lot.

**1:09:12** · I mean, it's not crazy, but I think the bigger problem is that many characters are actually rare, which means that it's really an inefficient use of a vocabulary.

**1:09:23** · And also, the compression ratio, which reflects this is not that great.

**1:09:29** · So most of the time, you're actually using a lot of tokens to represent your sequence.

**1:09:35** · And many of the indices are actually not being very used.

**1:09:41** · So this is not a very good tokenizer.

**1:09:45** · So here's another attempt.

**1:09:47** · So you can turn strings into bytes.

**1:09:50** · So Unicode has a UTF-8 encoding, which means that you can-- sometimes a string like "a" is just 1 byte, and sometimes, a string is multiple bytes.

**1:10:05** · So let's build a tokenizer around that.

**1:10:08** · So we can take this string and convert it into a sequence of bytes.

**1:10:14** · And notice that this is a longer sequence now, but all the numbers are between 0 and 255, because that's what a byte means.

**1:10:26** · And the compression ratio is 1, which is not great.

**1:10:34** · So byte sequences can be very long, but the vocab size is small.

**1:10:44** · OK.

**1:10:45** · So both of these are really bad.

**1:10:48** · So let's try to make some progress.

**1:10:50** · So this is what actually people used to do in NLP, if people remember.

**1:10:55** · So if you take a string, I can just chunk it up into-- break it up by spaces or some regular expression.

**1:11:04** · And let's just call each of these chunks a token.

**1:11:09** · So what is good about this one is that each token is meaningful because humans invented words, and words tend to have a stable semantic meaning.

**1:11:19** · But your vocab size is the number of distinct chunks in the training data, which could be a lot.

**1:11:25** · And also, your compression ratio, I mean, it's quite good, but the vocabulary can be huge.

**1:11:32** · Actually, it's worse than that because though the vocabulary could be actually unbounded, right?

**1:11:40** · Because at test time, you might get some sequence and you tokenize, and then you have a token you've never seen before.

**1:11:47** · And people used to assign these UNK token, but that's really ugly and can mess up your perplexity calculations.

**1:11:55** · So this is also not great.

**1:11:58** · OK, so what we're actually going to do is called byte pair encoding.

**1:12:03** · And this was introduced a long time ago for data compression.

**1:12:06** · Way before language models were really on the scene, really.

**1:12:12** · It was first introduced to NLP for doing neural machine translation.

**1:12:18** · And the first paper that used BPE for LLMs was GPT-2.

**1:12:27** · So the basic idea is that you're going to train the tokenizer on raw text to construct a vocabulary that's tailored to the data.

**1:12:36** · And you're also going to have this property that everything becomes-- can be tokenized.

**1:12:39** · If it's rare, then it just breaks up into smaller units rather than having this UNK token.

**1:12:46** · So common sequences are going to be represented as one token.

**1:12:51** · Rare sequences are going to be split into multiple tokens.

**1:12:54** · That's the idea.

**1:12:55** · So the algorithm is fairly simple, conceptually.

**1:12:59** · So you start, basically, with your corpus.

**1:13:02** · Let's assume it's one long sequence.

**1:13:05** · You get a byte sequence.

**1:13:06** · Each byte starts as a token.

**1:13:08** · And then we're going to merge successive pairs of adjacent tokens that occur the most frequently.

**1:13:16** · So let's step through how this is going to work in code.

**1:13:20** · So here's a simple string, "the cat in the hat," and here's the implementation of the BPE algorithm.

**1:13:28** · So we're going to turn that into a sequence of bytes.

**1:13:33** · And then we're going to-- first, we're going to, basically, count the number of times successive tokens appear.

**1:13:42** · So 116, 104 shows up twice, so we get this.

**1:13:48** · And then we're going to find the pair that happens the most number of times.

**1:13:56** · So that's 116, 104.

**1:13:57** · I guess there's a few ties, but we'll just take the first one.

**1:14:01** · And then we're going to merge that pair.

**1:14:03** · And by merging that pair what we do is we create a new token.

**1:14:08** · In this case, this is going to be called token 256.

**1:14:13** · It's going to represent this pair, and we're going to add it to our vocabulary.

**1:14:19** · So 256 is going to represent the sequence of th.

**1:14:23** · So t and h have been merged.

**1:14:25** · And we're going to call-- every time we see th, we're going to use 256 to represent that.

**1:14:31** · And then we go through indices, and then we replace every occurrence of 116, 104 with 256.

**1:14:37** · So those two places have been replaced.

**1:14:40** · And then we iterate.

**1:14:42** · So the next time we do this, we're going to find 256 and 101.

**1:14:48** · We're going to merge that.

**1:14:50** · And now we have 257.

**1:14:52** · And then we're going to merge that one more time, and we're going to get 258.

**1:14:58** · So over time, the sequence is shrinking and the vocabulary size is growing.

**1:15:06** · OK.

**1:15:07** · So let me-- and the compression ratio here is-- that we get for this, I mean, this toy example is 1.5 OK.

**1:15:21** · So now that you have a tokenizer, how do you tokenize new text?

**1:15:27** · Well, you take a new string and you encode it.

**1:15:31** · And conceptually, what happens is that you basically go through the set of merges that you've made, and then you just apply the merges to your string.

**1:15:44** · OK, let me actually not step through that code.

**1:15:48** · So that will give you a sequence.

**1:15:52** · So this is a sequence encoding of "the quick brown fox."

**1:15:56** · And then when you decode it, you get the same thing back.

**1:16:00** · So I went through this a bit fast, just in the interest of time.

**1:16:06** · I will say that this implementation works.

**1:16:11** · This is a full-blown BPE implementation.

**1:16:13** · It's extremely slow.

**1:16:15** · So in Assignment 1, we're going to ask you to, basically, make it faster.

**1:16:22** · So currently, encode loops over all the merges, which is very slow because you might have-- the number of merges you have is essentially the vocab size minus 256.

**1:16:37** · So you only want to loop over the merges that matter.

**1:16:41** · And you have to build some indices to make that happen.

**1:16:44** · There's some details around special tokens.

**1:16:48** · Conceptually not deep, but important to building a modern tokenizer.

**1:16:52** · Another thing is that I've presented the tokenizer just for simplicity.

**1:16:56** · As you take an entire string and then you try to tokenize it, really, what happens is that you break it up into-- your text into chunks, and then you apply tokenizer on each chunk.

**1:17:08** · So that's going to be much faster.

**1:17:13** · And then try to make it as fast as possible.

**1:17:16** · At some point you might realize that Python is just not very fast.

**1:17:20** · And if you want to implement it in your favorite language, Rust or C or something, then go for it.

**1:17:29** · OK, so quick summary.

**1:17:31** · Tokenizers convert between strings and tokens or indices.

**1:17:37** · The previous character-base, byte-base, word-base are highly suboptimal in their own way.

**1:17:44** · BPE is effective heuristic that is data driven.

**1:17:48** · So it seems to be pretty effective.

**1:17:53** · Now, like I said before, maybe next year, I don't have to teach this, but for this year, we're stuck with tokenization.

**1:18:01** · Even if we get rid of tokenization though, I think whatever solution replaces it, I think, has to satisfy the following properties.

**1:18:11** · If you have the model, the transformer needs to operate on some sort of abstractions of the sequence.

**1:18:18** · And this is most evident if you think about not just text, but video or DNA sequences where the individual bytes or units are actually quite low signal to noise, and you have to do some sort of abstraction to lift it into a place where you can do modeling on that.

**1:18:41** · And then finally, as I mentioned, chunks should be variable.

**1:18:45** · You want adaptive computation.

**1:18:48** · Not all bytes are treated the same.

**1:18:50** · And if you don't do that, I think, you're going to be suboptimal.

**1:18:53** · So any end-to-end solution also, I think, has to have these properties.

**1:18:58** · OK.

**1:18:59** · So with that, I will end.

**1:19:01** · Next time, on Wednesday, we're going to start the unit on resource accounting.

**1:19:06** · Which is sort of a baby system, I would say.

**1:19:11** · And then after that, we're going to go back into architectures and go from there.

**1:19:15** · All right.
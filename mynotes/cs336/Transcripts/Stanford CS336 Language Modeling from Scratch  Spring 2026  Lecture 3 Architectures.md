---
title: "Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 3: Architectures"
source: "https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=3"
author:
  - "[[Stanford Online]]"
published:
created: 2026-09-05
description: "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."
tags:
  - "clippings"
---
![](https://www.youtube.com/watch?v=lVynu4bo1rY)

## Transcript

**0:05** · So today, we're going to talk about architecture, which, at least to me, has always been pretty inscrutable.

**0:12** · And so I'm going to take the approach of just telling you everything. I'm going to go through all of the modern papers.

**0:19** · And we're going to just look through what has everyone done. And so I've titled this everything you didn't want to know about architectures and hyperparameters, because I think we all wished we lived in a world, where the only things you had to know were like VC dimension or something, very simple theoretical tools, but that's not really where we are. OK.

**0:37** · What we are going to do is, we are going to try to understand architecture from a survey lens.

**0:45** · The best thing to do, better than listening to this lecture, even, is for you to go out, and train your own models, and try different architectures. That's, by far, the best thing to do. That's part of the philosophy of the course.

**0:55** · But we're not going to be able to cover the whole design space of all the different architectures that are out there. That's not something that we have the compute or the time to do. So my opinion is, the second best thing that we could do is to try to learn from the experience of others. What has everyone else done? What are the choices that they are making?

**1:13** · And by looking at a broader, somewhat zoomed out picture, maybe we can start to understand, oh, these are the kinds of parameters and choices that are fixed across all effective architectures.

**1:24** · And these other ones can be varied without impacting how the model perform. So I'm going to talk about, basically, transformer variants, like, what is the modern transformer starting with the Vaswani paper?

**1:37** · And then as we go to more modern, more recent architectures, what do they have in common?

**1:42** · And then what are we allowed to vary or not allowed? But what do people vary as they go through this. So I think many of you have taken an NLP course of some kind or at least seen a transformer. So you've probably seen the very vanilla transformer from Vaswani, et al.

**1:59** · There's some fairly standard choices that you make. You say, oh, transformers don't have positional dependence, so we're going to add a position embedding. And what do we do, we're going to add some sines and cosines. We're going to have information processing through ReLU.

**2:15** · And then we're going to have a post norm. I'll talk about what exactly that is later. And when you look at your assignment, your A1, you're going to notice some differences between the standard or the vanilla transformer and what we've asked you to implement.

**2:28** · Well, we're going to ask you to move the LayerNorm to the front of each transformer block or the nonresidual layers.

**2:33** · We're going to ask you to implement something called RoPE. And we're going to ask you to implement something called SwiGLU GeLU and not ReLU. Why do we pick these? One reason is, we've copied a lot of this over from but Llama, so did everyone else. Really, I think, if you were to train your own language model, I think you'll quickly run into this question, oh, there's so many choices. What do I choose for all of these things? And so let's now walk through all of these different models?

**3:01** · The way I think about architectures is to look at all the different things people have done and say, what are the things that people have done? Can we pick and choose from those?

**3:11** · Percy always makes fun of me for this a little bit, but I try to look at all the different models that come out each year to try to make this lecture.

**3:19** · And last year, I thought, oh, there's just a couple papers. It's going to be fine. It's going to be fine. And then I look through all the things, and there's a lot of papers.

**3:25** · There's Qwen2, and Gemma 3, and InternLM2. And then there were even more.

**3:31** · There's Nemotron-4 and Qwen2. And oh, my goodness, there were 19 new dense models. And so last year, I had my work cut out for me.

**3:38** · And this year, I thought, well, there can't be that many new LLM releases. It's got to be slowing down.

**3:44** · People can't keep training 20 dense LLMs per year. And that's technically right, there aren't as many dense LLMs.

**3:50** · Initially, I was like, oh, there's Qwen3. Gemma 4 just came out last Thursday, so I put that in there, and Olmo 3.

**3:56** · There's only a couple. And of course, I have to give a shout out to Percy's own AP model trained with Marin.

**4:01** · And I was like, oh, we'll just have a few things to cover. But it turns out, if you start looking, there's a lot of different models.

**4:07** · And so the fact that we have so many different models, most of these actually are MoEs, mixtures of experts.

**4:13** · And I'll be talking about that tomorrow rather than today. Because we have such a big diversity of models, we actually get a pretty good picture of all the different choices that we can make. So I made this little table. We'll come back to this little table at the end of the lecture.

**4:29** · But basically, at this point, starting with the original transformer, there's been actually quite a few autoregressive language models trained on the same class of things. And you can ask questions like, what are the different vocabulary sizes? Or what kind of layer norms do we use? Or what kind of position embeddings do people use?

**4:49** · And we see some fairly clear trends. I'll be talking about this as we go. So the goal here is that we're going to cover a couple of different things. We're going to cover common architecture variations. So these are different building blocks of the transformer.

**5:05** · And after we've established what the standard building blocks are, like, what do we use for the nonlinearities?

**5:11** · Or what do we use for position embeddings? Then we're going to talk about hyperparameters. We're going to go down even lower detail and say, what is ff\_dim? Should we make that a multiple of four or multiply the hidden by four to get ff\_dim?

**5:25** · How many vocab elements should I have? And then after that, we're going to talk about very low-level tricks of how to get models to train stably. And the reason why I'm going to talk about that in this lecture is because these stability tricks have a pretty close connection with the architecture variation. One of the things that higher level I want to impress upon you is that architectures are actually a very complex set of trade offs. What does the architecture have to do?

**5:53** · Well, it has to learn from data, so it has to generalize. It has to train efficiently on GPUs.

**5:58** · And it has to not blow up. Halfway through training, if your training loss is just go down like this and then suddenly blow up, that's no good at all. So all these different requirements end up getting baked straight into the architecture. And that's why these things are a little bit messy and a little bit complex.

**6:16** · But you should keep that in mind. And that's why things are, in many ways, not so elegant. So we're going to start with the core architecture piece.

**6:25** · And as a high-level view, I think the way that I see a lot of the architecture stuff, looking, basically, historically, is in the early days of starting with a transformer until GPT-3 or so. There's a lot of experimentation that happens. People try lots of different things. There's no gold standard that everyone has unified on.

**6:45** · And then Llama 2 comes out. And everyone's like, wow, Llama 2 is great, I want my own Llama 2.

**6:51** · And so everyone starts training Llama 2 alikes with minor variation that people have.

**6:57** · And then finally, last year, we saw really big differences or a trend towards architecture modifications that make training more stable. And this year, we see lots of trends towards architecture variations that enable longer context dependence. So there are these big themes that are happening. But really, I think you see this big point when Llama 2 comes out, and everyone's like, wow, I want to train some of that. And then suddenly, or not suddenly, but after that, people are starting to explore once again. So it's cool to see all of these different changes.

**7:29** · OK. I think people can disagree about a lot of things on architectures, but there is one thing that everyone agrees on. If you take the transformer paper, I think a lot of people say like, the transformer people got most of the things,

**7:44** · except this. And the thing that they really did not get right or I think, most people agree, they did not get right is where you put the layer norm. So in the original transformer paper, the layer norm goes in what you would call the residual path. So in the transformer, you have the residual stream that's x, that runs through the whole network. And then you are going to be applying attention, which does some computation. And it will add back a delta back into the residual stream.

**8:15** · And then, in order to make sure that these gradients are stable across layers, a layer norm is placed at the end of each of these components. Now, instead of putting the layer norms in the residual stream, there's an alternative, I'll refer to this as prenorm, in which you can put the layer norm outside of the residual stream, but before each of the computations. So you can put it before the multihead attention, you can put it before the FFN. We'll call this prenorm. The nomenclature will get a little bit confusing.

**8:49** · You can call this postnorm for now, but let's call this residual norm, because you're putting the norm in the residual layer.

**8:56** · Basically, all modern language models push the layer norm outside of the residual stream.

**9:03** · This is just like a thing that basically, everybody does. There is one funny exception, but it is OPT350M.

**9:11** · And if you all are familiar with language models, we know, OPT, in general, was a mess of a language model.

**9:19** · And OPT 350M is even more so because I don't why only that model has a postlayer norm in the residual stream.

**9:28** · OK. So this is one of the things that everyone agrees on. And so you might wonder, why is this such a unified thing across all of the different models? And if you look at some of the early work, studying, where do you place the layer norm style research, what you really see is that the early motivation for a lot of this was, when you train a transformer, you need to do a warm up. Actually, modern transformer training still does warm up as well. But you definitely need to do warm up when you train. Now, wouldn't it be nice if we could remove the warm up.

**10:07** · So that was the initial motivation for a lot of this research. But people quickly realized that removing the warm up had very serious issues in terms of the stability and convergence of these things.

**10:19** · So if you did postnorm plus layer norm, which is basically the original transformer thing, you got this purple dashed line, oh, you just don't converge as well compared to doing something like prenorm, you can ignore the other terms, you would get much nicer convergence even without warm up. So this was the original motivation.

**10:38** · But really, what people quickly realized is that moving the layer norms outside the residual stream has some pretty important implications as you make your network deeper and as you start to grapple with stability issues. To me, I think the gradient attenuation issues are the most clear.

**11:00** · When you talk to people who do architecture design, I'm not really one of the people that deeply engages in this.

**11:06** · But one of the things that people often say is, keep your residual stream clean. So in this case, you have your x's coming in from the on, the prenorm side. And this x propagates all the way up to the top, all the way up to your final output. And that allows gradients to propagate in the backward pass, straight through this. That makes gradient propagation very simple, which improves both stability and signal propagation. And that's what people realized very, very quickly, that if you do something like prenorm in blue at initialization, the gradient sizes remains the same,

**11:43** · because you have this nice, straight through propagation in the backward pass. On the other hand, if you have post LayerNorm, you have these complicated effects that happen, because your LayerNorming, each time, you're going through a transformer block. And that's going to change the norm of your gradients as you go backwards through.

**12:00** · So you can see, from the principle of keep your residual stream clean, that prenorm makes a lot of sense. People also realized through experimentation that this also improves stability, in general, that the sizes and frequencies of gradient spikes were improved under prenorm compared to postnorm.

**12:21** · And this is a figure from Salazar and Ngyuen, who were one of the first ones, I think, to study this phenomena carefully.

**12:28** · I think this is the reason why it's stuck around. Stability and the ability to go deep are both very, very important for modern large language models. And so this idea of moving your LayerNorm outside of the residual stream is one that basically everyone has adopted. So now, if putting LayerNorms in residual streams is bad, why does LayerNorm have to be at the start? Of course, we have prenorm, which is before our computation.

**12:57** · But we could have it after computation as well. That's equally good, at least under that logic.

**13:03** · And that's exactly right. Many recent models, like Grok, or Gemma 2, or OLMo 2, have this structure where they've moved the LayerNorm after the computation. So it's a postnorm of a kind, but it's outside the residual stream. Other models still actually just put LayerNorms everywhere. They put a LayerNorm here.

**13:21** · They put a LayerNorm after. I'll get to this later as we talk about stability. But one of the other lessons that seems to have held up very well is, if you have stability issues, you can sprinkle in LayerNorms everywhere.

**13:33** · And that will generally improve stability. It's almost very strange to be saying this, because it's so ridiculous.

**13:39** · And yet, that statement has actually been proven right. Every time people have encountered stability issues, they say, oh, but what if we just throw a LayerNorm into attention? Turns out, that works too. We'll get to that later.

**13:49** · OK, that's postnorm or double norm, in this case, where you have two LayerNorms here.

**13:55** · OK. The other thing that you can do is, in the original transformer, you have the layer norm, which is this operation right here.

**14:02** · So you have your activations x, you're going to mean subtract, divide the variance, and then scale it back up.

**14:08** · And this works just fine. It's not like this is wrong. And many models have successfully trained on this scheme. But basically, most or all modern models, I think, use RMSNorm, which doesn't subtract the mean or add a bias term. So it's just a scaling down and scaling back up.

**14:26** · So you can see this in the equation here. And really, LayerNorm is more expressive than RMSNorm.

**14:33** · So there's really representationally no reason why you have to use RMSNorm.

**14:38** · But RMSNorm is nice, because in practice, there's really no expressiveness loss.

**14:44** · RMSNorm models just as well as LayerNorm. But more importantly, it is faster.

**14:50** · This is the part where the systems and architecture codesign starts to come in.

**14:56** · Firstly mentioned in the previous lecture, this idea of arithmetic intensity. We want to keep our GPUs hot by doing matrix multiplies and other very intense computations. We do not want to be wasting our GPUs by having them move little tiny bits of memory back and forth. That's a very inefficient use of our very powerful GPU.

**15:16** · And so what we want is to remove operations that are small and involve memory movement, but don't give us much expressive power. So by that view, what we really want to be doing here is, if the mean subtraction and addition isn't really doing much for us. Just get rid of it. You might think, OK, why does this matter?

**15:37** · We're just optimizing this teeny tiny operation that accounts for, in this case, something like 0.17% of the total floating point operations of our system.

**15:49** · But as Percy mentioned, it's not really about the flops. The flops are the floating point operations we do, that's multiplying matrices, but that's not runtime. Runtime is a much more complicated object.

**16:02** · And statistical normalizations, things like LayerNorm, even though they're only 0.17% of the flops, depending on your workload and depending on the setup, can be up to 25% of the runtime. That's crazy.

**16:15** · On tiny models, this can be really, really big, because you're still having to move all these parameters back and forth from fast to slow memory and vice versa when you're doing these operations. So data movement is really, really important. And RMSNorm can still matter a lot because of this.

**16:31** · So you can see the difference here. The arithmetic intensity is in white.

**16:37** · And then you can see the flops involved in black. And you see that LayerNorm has a very low arithmetic intensity, which is the operation we try to want to remove as much as possible. Yeah, question over there?

**16:48** · The movement for normalization is so disproportionate compared to density \[INAUDIBLE\].

**16:54** · So for something like tensor contraction, which is, in this case, matrix multiplies, the majority of the workload is multiplying, whereas for stat normalization, the majority of the workload is memory movement. And memory movement is quite slow. So imagine the case, where moving something is almost all the compute. Then you're still paying quite a bit here, because activations can be quite large.

**17:18** · Yeah, I think the percent runtime in this case is quite extreme. This is like in the tiny models with matrices that don't really generally make sense in modern workloads, but this is giving you a sense of why this is a free optimization.

**17:31** · And you do see this. This is another paper in which people were evaluating different architecture interventions, Narang et al in 2020. I think this was a Google paper. And they show, for teeny tiny transformer of a 200 million parameters, you get more steps per second. That's the third column over here when you switch to RMSNorm.

**17:48** · And in fact, you actually get better performance, which I don't think is something that you're guaranteed, but it's a nice bonus, regardless.

**17:54** · So you get a free systems win by just moving to RMSNorm. And so basically, everyone has decided to move over to this now. And in general, there's a more general version of this.

**18:06** · There's no guarantee to any of the things I'm saying. But biased terms in transformers and neural networks are generally not that useful. So in the original transformer, the linear terms all have biases, but most implementations actually just drop the biases entirely. Once again, this is another example of something that's not very arithmetically intense, but fairly memory intensive, relatively speaking. And so you might as well just drop these and get the free systems win.

**18:35** · There's also some cases, I'll just mention this offhand, where the bias terms can also induce stability issues, so they're useful in other ways.

**18:42** · But really, I think the primary reason these are dropped is just to simplify things from a systems perspective.

**18:49** · Cool. OK. So I think layer norms, the story is pretty easy.

**18:55** · It's easy in the sense that what people do is fairly standardized. Our understanding, not like a deep theoretical level, but our understanding of what layer norm does is fairly good. Everyone moves the layer norm outside the residual stream, often prenorm. But I think this might partially be because Llama 2 did that.

**19:16** · And we roughly have a sense of how to use LayerNorm to control things like gradient spikes and keep signal propagation nice. Related to that, we also now basically always use RMSNorm.

**19:29** · And you hopefully understand the general principles here of basically, just dropping bias terms.

**19:34** · And that allows us to keep our system more arithmetically intense while keeping the expressive power the same.

**19:43** · I think the unsatisfying thing about a lot of architectures is that, you can't really reason about this beforehand, but we don't beforehand that dropping the bias terms is OK. But from a lot of experimentation and now, collectively acquired knowledge, we roughly that dropping the bias terms on both the linear and RMSNorm is OK for typical language modeling workloads.

**20:02** · This is the kind of statement that we can make on the basis of what we do when we look at a variety of different models.

**20:09** · Any questions for LayerNorm stuff? Good. OK.

**20:15** · So now, I'm going to talk about activations. And there's a whole zoo of activations.

**20:20** · There's just a lot-- ReLU, GeLU, Swish, ELU, GeGLU, SeLU, SwiGLU, LiGLU.

**20:27** · And what are these things? I think, at one point of my more stats ML training, I thought to myself, I will never learn these things. I will make it a point of pride to never know what a SwiGLU is.

**20:38** · But now, it's actually very important for us to actually have a general sense of what these objects are and which parts of these names actually matter for performance. So you can build and train a language model on just

**20:55** · a fairly vanilla activation. I guess, Chinchilla is probably the best model out of that group. But even if you just want a ReLU, you can train a reasonably performant language model using just that activation. There's nothing wrong with that. And if we move to GeLU, which is a Gaussian error unit, and really, the only difference is this tiny divot at the bottom here, which really, for the most of the activation doesn't change anything but changes the gradients right near zero. Then you can train models like GPT 3.

**21:25** · That's a perfectly good large language model, not modern by modern standards, but perfectly fine.

**21:32** · But then we get to the gated linear units, like SwiGLU and GeGLU. And these are really where most of the action is.

**21:39** · This is very similar to LayerNorm in that, I think almost all credible modern language models use a gated linear unit of some kind. OK. So what is gated linear unit?

**21:52** · So these are gated activations. So if we want to look at something like a feed forward layer, we can just look at this first part. This is a very standard ReLU feed forward. I have my x. I hit it with a W1.

**22:05** · I entrywise threshold at 0. And then I hit it with another W2. I get my output right. Very straightforward ReLU network.

**22:11** · I don't say this as my personal experience, but another thing that is often said in architecture design is that gating is often very helpful.

**22:21** · So if you apply that very general heuristic, what you might get is to say, OK, well, instead of just having this entrywise ReLU, why don't we also have a gate? And the second gate, the second term here, this is just going to multiply the output of my ReLU, entrywise. And I have a second matrix V. OK.

**22:42** · Now, this is just going to modulate the output of my ReLU. And then I'm going to do everything else the same.

**22:48** · So instead of just having xW1, W2, I have xW1, and I'm going to gate that with xV. This is another activation the same size as this.

**22:59** · And then I'm going to down project it back with W2. OK. So what is this?

**23:04** · Now, this is a ReGLU. You make these names by adding the first activation, in this case, ReLU and GLU, so the ReLU gated linear unit.

**23:14** · And gating has been a very effective other primitive in architecture design.

**23:19** · And it turns out that this is very effective in language modeling as well. So if you take something like a GeLU, we've already talked about that. That's the ReLU, but with a little divot at the bottom here, you will get a GeGLU.

**23:33** · And if you take a SwiGLU, which is x times a sigmoid, then you will get a SwiGLU.

**23:39** · So this is a squish times the rest of it. And this really covers a lot of the modern models.

**23:47** · Generally, the Google folks have used GeGLU, so like the Gemma models, the T5 models are those.

**23:53** · And everything that's a Llama descendant uses a SwiGLU. So PaLM and the Llama descendants are all SwiGLU models. I would say that SwiGLU is probably the more dominant one, but honestly, amongst the gated units, doesn't really matter. Now, here's a side note that will be a semi-important piece of trivia later. If you look up here, you will notice that there are more parameters for the gated model, because I have this parameter V. And so if you do a little bit of math,

**24:27** · I now have three matrices instead of two matrices. What you should do is, you should maybe use a smaller feedforward dimension by a factor of 2/3, in order to keep the parameter count the same.

**24:40** · So this is roughly the idea of, well, I want to keep the same number of total parameters as my original MLP, but I now want to make it gated.

**24:48** · So I'm going to make the feedforward dimension, which is the output dimension of this W, a little bit smaller by 2/3.

**24:54** · So this is a general rule of thumb that people have followed, but it's not really an iron rule.

**25:01** · The original Noam Shazeer paper that proposed this had some very small deltas originally, but they're consistent deltas. And I think to his credit, I think a lot of his papers have these error bar assessments of training multiple replicates and checking to see if they're better.

**25:22** · And if you look, the GLU variants are almost always consistently better than the nonGLU variants.

**25:29** · And this is a parameter matched comparison, because Noam Shazeer is always doing this 2/3 adjustment to make sure that all of the models have the same total number of parameters.

**25:40** · So this is quite nice. It's, in some ways, a free win. Almost everyone uses a GLU.

**25:46** · There have been other more controlled systematic comparisons. This is the same paper I was talking about before, Narang et al in 2020. Google, actually, in the 2020s, did quite a few nice large scale architecture comparison papers, although with a T5 architecture and not an autoregressive language model.

**26:07** · And they basically comprehensively compare things like GLU. And you see, once again, if we look at the SwiGLU, or the GeGLU, or the GLUs in general, they do significantly better at loss or the other downstream metrics. Fairly compelling on these papers, also clear from now a lot of model training runs that SwiGLU and GLU are good.

**26:30** · So there's a lot of variations in gating, but really, the important single axis to know is that gating for these nonlinearities is actually quite important, gives you a nice boost without much of a computational cost.

**26:44** · That's not to say that gated linear units are necessary. I mean, GPT 3 was that-- I think, the Nemotron 340B model used a Squared ReLU, which is a crazy choice, but that works too.

**26:56** · Both of these models are perfectly performant, but it's actually quite rare to see anything that's not trained on a gated linear unit. So evidence is pointing towards consistent gains on using these gating tricks.

**27:09** · So those are, I think, the more consensus choices for things that we can do in architecture. Now, this one, I think, is a really fun idea, but one that, I think, now, the test of time has shown maybe is not quite as good or maybe not as popular of an idea. Normally, we do our transformer blocks serially.

**27:29** · We compute our attention, then we compute the MLP, one after the other.

**27:34** · If you're very systems-minded, you might say, well, this introduces a bottleneck. I have to wait for the computation of one to do the other. If they were, instead, in parallel, I could bring to bear some new and cool systems optimizations, potentially. So you might ask, could we parallelize the transformer block?

**27:51** · And this was originally an idea that was in GPTJ, which is an open source attempted replication of GPT 3. And very interestingly, I think GPTJ has been surprisingly influential in propagating a lot of ideas.

**28:10** · I mean, PaLM as well. Google is actually surprisingly bold with the architectures that they do.

**28:16** · But the description in PaLM, which you can see in their report, is the following-- instead of nesting this, which is the sequential firm at the top, you're just going to add together the output of the MLP and the attention layer and just add both of those back into the residual stream.

**28:32** · If you implement this right, you can actually share a lot of the components. You can share the LayerNorms. You can fuse the matrix multiplies.

**28:40** · This allows you to potentially get additional systems optimizations. And I think a lot of the people that have been influenced by Google. So Cohere, I know, was founded from one of the former transformer authors.

**28:53** · They do a lot of Google-inspired optimizations. They followed this architecture, but not very many others.

**28:59** · This has been an approach that has really fallen out of popularity over the past, I think, two years, I think mainly because optimization of the serial form has gotten sufficiently good that the system's gains from the second one just isn't worth the small hits to representation power that you end up getting, going from parallel to serial. Effectively, you can think about it, as you've lost half of your depth. And that can be a deleterious thing to do to your model.

**29:29** · So in terms of the architecture things, actually, the fact that this is so short should suggest to you how much the original transformer formulation has somewhat stood the test of time, because the only thing I'm really talking about changing here is where the norms go, or whether we have bias terms, or whether we get the MLPs.

**29:51** · But those are actually pretty minor changes compared to all the things that you can do. Now, those of you that are carefully paying attention might say, but wait, there's a lot of transformer alternatives that change the attention.

**30:04** · Yes, you'll have to wait until next lecture, because today, I'm just only going to cover core attention-based methods.

**30:12** · And next lecture, I'll throw in a little bit of state space model stuff. But as long as you're in this dense attention land, actually, the architecture from the original transformer paper is pretty close to what we do. So you see quite a bit of this.

**30:25** · So just now going back to this, blue here is RMSNorm, black is LayerNorm. You see most of the modern models are RMSNorm models, serial versus parallel layers. The blue one is parallel, the rest is serial.

**30:37** · You see mostly serial layers, prenorm versus postnorm. Some of these ones that I marked as postnorm are actually pre and postnorm. And then these ones on the right, these are GLUs, almost always, with the exception of things like Falcon, which use a gated linear unit. But almost all of these are really gated linear units for modern models.

**30:59** · So you can see the trends quite visually from what I'm telling you. OK.

**31:04** · So really, the thing that is very different across implementations, and I think a place where a lot of the architecture stuff is still in flux is how you do position dependence and incorporate information from other positions.

**31:18** · So the core attention component, in some sense. So there are lots of different ways that you can encode position into a transformer. And just to remind you right, this is very, very important, because attention is positionally independent. They're just inner products, so you can just shuffle them.

**31:36** · And attention would be the same if you don't have a position embedding. The original transformer had sine and cosine embeddings, a Fourier transform intuition that if you have sines and cosines, then you can recover position from that, no matter what.

**31:50** · A number of other large models that followed soon after that used absolute embeddings, where each position had its own different embedding. And then several other Google models like to use relative embeddings. And here, you're not adding embeddings into the word vector embeddings, but instead, you're adding a vector to the attention computation itself. So if you're three positions off, the attention matrix gets a different offset added to it. And models like T5 and Chinchilla use this scheme.

**32:24** · The thing that has really become pretty dominant in terms of position embedding is this class of embeddings called RoPE, which some of you may be familiar with. Most models past 2024 use this type of embedding.

**32:37** · And it's remarkable, given that RoPE, in some ways, came out of nowhere.

**32:42** · Originally, I think this was also a GPTJ innovation from, I think, not very well-known blog post and paper combination from an author in China. But really, it has some really interesting ideas for why you would do something like RoPE. So RoPE is a relative position embedding.

**33:04** · And a relative position embedding, let's make an opinionated stance that I should not care about the absolute position of any words. So if you A and apple appear together, even if it appears at the start or at the end, in RoPE embeddings, they should get the same result.

**33:28** · And we want to represent it in this way. So I have an embedding f, and I have another embedding f.

**33:34** · And these are going to take in the identity of the words x and y and the positions absolute of i and j.

**33:40** · And I want this to be equal. If I take the inner product of these embeddings to be equal to a function, that only depends on the relative difference. And every existing embedding before it didn't really fulfill this equality. Sine is not relative, because it has these absolute cross terms that are not relative. Absolute position embeddings, just by the name of it, is obviously not relative.

**34:03** · And then relative embeddings, technically, these are relative, but they're not embeddings, because they're just adding to the attention matrix. So there's no inner product structure that you can extract out of this guy.

**34:16** · So given this, you might ask, is there a nice way that we can truly have this relative embedding?

**34:23** · And the idea is very cool. It's really just looking at other properties about angles and cosines.

**34:30** · So we want our embeddings to be invariant to absolute positions. And we know that inner products of any kind are invariant to arbitrary rotation. So the idea is to say, I'm going to take my semantic word vectors, the ones that are independent of any position. So this is my starting point. And then I'm going to rotate each of these vectors, in this case, in 2D, based on the position that the words appear.

**34:56** · So just as a simple example, let's say we have the sentence, "We know that." "We" appears as at position 0, so I'm not going to touch that at all. I'm just going to keep that where it is.

**35:10** · The word "no" is at position 1, so I'm going to rotate it by some angle. And that's my one position rotation.

**35:18** · Now, what happens if I apply the same idea to the following sequence?

**35:23** · Of course, we know. In this case, "we" and "no" are still adjacent. They're right next to each other, but their absolute position has shifted.

**35:30** · "Of course" comes before "we know now." In this case, I'm going to rotate the word "we" by two positions, because it's two index, 0, 1, 2. So the word "we" is in the second position number 2, so I rotate by two. I rotate "no" by three positions, because it's in position number 3.

**35:47** · And what do you know the relative angle between these two is still separated by 1. So this is a very, very simple idea of just using rotations to represent position. And if we do that, then anytime we take an inner product, those inner products are going to be invariant of absolute positions. Now, you might say, well, in two dimensions, that's pretty easy, because you've only really got one choice, you've got clockwise and counterclockwise. But in high dimensions, there's an infinite space of ways that you can rotate vectors. So what do you do in D dimensions? Well, you do the simplest possible thing, and it works.

**36:23** · The simplest possible thing is to reduce it to the 3D case repeatedly. So you have a D dimensional vector.

**36:28** · Just cut it up into chunks of two. And each pair of two dimensions gets rotated.

**36:35** · And the theta at which these things rotate vary. Some of them are very low frequency, so they rotate very slowly, so they can capture long range dependence. Some of them rotate very quickly, so they capture things like, are they neighbors to each other?

**36:48** · And then at the end, after I've rotated every pair of vectors, I get my final embedding. So this is the RoPE approach.

**36:58** · The paper, if you read it, has a very complex motivation about complex numbers, but really, I think the intuitive way, at least to me to think about it, is to just, you want to rotate by reducing to the two-dimensional case, and you're just rotating every pair of coordinates.

**37:15** · Gemma 4 just came out on Thursday. And they have another different kind of fun thing that they do, which they call, I think, proportional RoPE or p-RoPE, which is a really strange way to just say that the only thing they rotate is the first two coordinates, but that's another valid thing that you can do as well. So there's a lot of different things that you can do in this space that end up working.

**37:35** · OK. In practice, what you're going to end up doing is, you can take your vector, and you can make a sparse multiply with sines and cosines. And this is going to be giving you some way of rotating your input vectors x's. So x times w times r, this is going to be your final embedding that you get. And finally, this is a sine and cosine, which looks a little like sine embeddings. But it's really important that I'm multiplying with these sines and cosines rather than using them as embeddings,

**38:07** · because that means that there are no cross terms. And this is purely relative. There's no absolute position information that you'll get out of inner products. If we really, really wanted to get into low-level details, and you ask, how do I actually implement this thing? You're going to have to do that. You have your usual attention stuff.

**38:24** · And then what you do is you generate cosine and sine angles based on the position IDs of where your sequence is.

**38:31** · And then you're going to apply those cosines and sines onto both your queries and keys for your attention computation. And you can either apply them as a matrix multiply, or you can go through and apply them manually, just as a rotation. Fairly straightforward. And you would do this at the attention level rather than at the very bottom, to enforce position invariance, every time you're doing attention computations. So that was RoPE.

**38:55** · It is a little bit confusing, but once you understand the geometry of just rotating things, it's actually fairly straightforward.

**39:02** · OK. I'm going to pause here for one moment, in case anyone has questions about the various architecture bits. We're going to then talk about even lower level details about hyperparameters, so yes.

**39:16** · Do you know of any papers that do a higher dimensional? Higher dimensional rotation.

**39:22** · It's a good question. I don't think so. But a higher dimensional rotation, like any 2D rotation in the space would just be a variant of this. You could certainly do any one manifold that is a closed loop. I have not seen that.

**39:36** · Yes. \[INAUDIBLE\] What do you think is the best way to distill this kind of knowledge from papers and technical reports? It's a good question.

**39:47** · I don't know if there's a way beyond some combination of looking broadly enough to get a pattern, which is what the procedure I'm trying to do in this lecture here. And then the other one is to try it yourself, even a much smaller scale, to form an intuition in a theory for how these things come together. I think those two are really the right ways.

**40:06** · I think reading any single paper in isolation is very, very difficult, especially now, because no single paper seems to give any full detail for a lot of language models these days. Oh, lots of questions now. OK, good.

**40:19** · We'll go in. Yeah. So I have a question on the parallel layers and \[INAUDIBLE\] layers.

**40:25** · Yeah, I understand, the modern models are thinking of the resource efficiency, so they use the parallel layers.

**40:32** · But there's a difference between accuracy for these two patterns.

**40:39** · I want to know what's the degree of accuracy. Is it big enough or also small enough to allow the current model trainers to ignore that, or \[INAUDIBLE\]? Yeah, I think that's actually really mixed.

**40:54** · So if you read the original PaLM paper, I think they're very confident about the use of parallel layers, like no performance drop, 15% systems utilization improvement. So if you read just that, you'll say, oh, it's just as good.

**41:06** · But I think a lot of the later Google models have stopped using this, which you can take on as an implicit signal that actually, there might be some losses. And once again, this one is a little bit hard to get precise numbers on, because no one's done the ablations, as far as I know on parallel versus serial, controlled nice ablations, at least.

**41:25** · Yeah. What's the difference between P-RoPE and RoPE if you might have read \[INAUDIBLE\]? Yeah. Yeah. I mean, this difference is really just, which of the coordinates you rotate?

**41:39** · You don't rotate most of them, because a lot of the-- I mean, the argument originally, I think, is that the low frequency parts just aren't rotating very much. And so you can drop them if you're really strapped for of extra space. And this is really an optimization for teeny tiny models, where you don't have very much hidden dimensions to have activations for. Yeah.

**42:03** · So the relative embeddings, not having an inner product.

**42:08** · Is that because it only applies to keys basically? I was trying to understand. Yeah. So they applied both to the keys and values, which is why you get this relative effect from where you are.

**42:21** · You want to not have cross terms. So if you look at the sine and cosine embeddings, then you'll not only get the original vectors, you'll get these weird cross terms between the position embeddings and the word embedding themselves, and so on and so forth. And then you can back out what the absolute position is. So even sine and cosine embeddings are not like pure relative position embeddings. You have to accept the premise that the relative embedding is what you want. But once you do, you end up at the RoPE solution somewhat naturally.

**42:53** · \[INAUDIBLE\] Yeah, Yeah.

**43:00** · So what's the issue? So the issue with this is that it just can't be factorized as a product. That's more of an aesthetic problem.

**43:06** · If your constraints are, I need it to be relative, and I need it to factorize as f of xi and f of yj, then this is not a solution in that class. To be fair, there's a lot of embeddings that work this way that do work, like l of i and other kinds of approaches do this inject into the attention matrix. And they do reasonably well.

**43:30** · It's not necessarily the one that's become the dominant approach, is what I can say.

**43:36** · Cool. OK. Great. Now, we'll talk about hyperparameters.

**43:43** · And I think hyperparameters are really something that you start to engage with once you actually have to train a model.

**43:48** · When your knowledge about language models are abstract, you don't have to care about any of these. But once you have to instantiate it, you start to ask questions like, well, how big should the feedforward size be? How many heads should I have?

**44:01** · What should my vocab size be? And you might also have questions of, what should my weight decay or dropout be?

**44:08** · Do I even need to regularize? I have a lot of tokens. So do I need regularization?

**44:14** · And do I need very deep models or very wide models? What are the right kinds of things to do here? And all of these, if you start out with no knowledge, it's actually very daunting, because you have to search this very big high-dimensional space.

**44:27** · But the space of things that people try is actually pretty small. And from that, maybe you can start to think about smarter search processes of where you want to vary things. One of the things that's a really consensus hyperparameter is this idea of the ratio between the feedforward size, which is the output of your first matrix multiply in an MLP and the model dimension. So this is really the ratio of the two dimensions of your W1 as well your W2 matrix. This seems like a thing that's very important and controls the richness of your MLPs.

**45:02** · So what should it be? Well, for whatever reason, it should maybe be four times your hidden dimension.

**45:10** · And this is a rule of thumb that works remarkably well. And I will show you some data on why maybe this is a fine number to choose. There's a few exceptions. And funnily enough, the really extreme exceptions backtrack on that.

**45:25** · OK. Exception number 1 is variance of the gated linear unit.

**45:30** · I already told you about this. So if you were thinking about it, this is probably cached in your head right. GLUs have more parameters if you keep the same dimensions.

**45:38** · So if you want to keep the parameter size of your MLPs the same, well, you need to scale down by 2/3.

**45:45** · So most GLU variants, this means that you're going to end up with something like 2.67-ish.

**45:51** · So everyone that's down here, 2.67 to 2.5, this is roughly applying this 2/3 correction.

**46:01** · And then for whatever reason, the Llama 2 folks decided, well, we actually have very efficient attention heads with MQA, which I'll talk about later.

**46:15** · And because of that, we can multiply this ratio by an arbitrary 1.33. And we'll get roughly 3.5.

**46:21** · And so the Llama people arbitrarily chose a slightly different ratio, which essentially emphasizes the MLPs a little bit more.

**46:28** · But really, if you actually look through all the papers, you'll find either 2.6-ish or 3.5 for GLUs, or 4 if you're doing non-GLU models.

**46:41** · OK. There's another exception, which I find to be very funny, but also very, very cool, which is, throughout, as you read these technical reports, you'll find that most people are just very boring in their choice of architectures. They're like, we did Llama, but we changed one thing.

**46:57** · But folks at Google are very bold, sometimes. And T5 is one of my favorite ones, because they have some really bold settings. They decided that instead of following this 4x rule of thumb, they decided that they want to have a 64x multiplier, which is way bigger than 4.

**47:17** · And they have a reasonable argument for this as well, this another systems-based argument. They said, well, the bigger my matrix multiplies, the more efficient I can keep my hardware. So if I make this multiplier really big, then my matrix multiplies can potentially be more efficiently utilized.

**47:36** · And some others, like Gemma 2, have also tried to really push a little bit higher on this.

**47:44** · But really, T5 is astounding exception at 64. I don't think any other model has really gone that high in the feed forward multiplier. And empirically, if you look at other works that try to do more controlled comparisons of this ratio, I've taken this one from Kaplan in 2020.

**48:03** · This is the classic neural scaling loss paper, where they do various controlled studies on language models. You'll see, this wasn't the point of the study. It was a scaling law study. But you'll see in one of the panels that they actually have ablation or sweep, where they change the feed forward ratio, and they look at the loss for a very small model here. But what they find in this paper is, there's a basin, where you start at about one, and you end about maybe 10, where this hyperparameter is pretty good and very, very flat.

**48:36** · You lose very little relative to the optimal loss down here. And then if you get it really wrong, you get above 10 to 100 or something like that, then your loss starts really shooting up quadratically. And so a lot of these choices that range between 2.6 to 4, they're all falling into this relatively nice basin, so you're fine choosing those numbers right.

**48:59** · So what can we learn about this hyperparameter? Well, the default choices have worked very well for nearly all modern language models. So you can safely choose that. T5 was a fine model or the version 1 T5 was a fine model.

**49:12** · It wasn't a bad model. And so even radical choices can technically work, but it's probably going to be compute-inefficient.

**49:20** · And I think the funniest part of this saga, this is the punchline of the T5 saga to me, is that they have a follow up model T5 v1.1 that's supposed to be the improved version of T5. And they go back to the standard 2.5 multiplier.

**49:34** · So there's nothing explicitly stated here. But clearly, when they tried to update T5, they decided that they wanted to go back to a more standard multiplier, which I find to be a little bit funny. So that's the feed forward ratio, which now, you have a rough sense of what the right order of magnitude is. Now, let's talk about a different consensus hyperparameter. I always found this to be very strange when teaching 224n and just teaching students about this, which is, if you have a multihead attention, where you have multiple heads for your attention

**50:09** · in your transformer, the canonical thing to do, the thing that almost everyone does is, if you have multiple heads, you make sure that the size of those heads, the head dimension is such that you have the same dimension as a single head transformer. So you always make sure that you divide the hidden dimension to basically, multiply with h. So in this case, you have h the number of heads, and the dimension of each head is d over h.

**50:35** · So you multiply the two and you get d. For some reason, this is the rule of thumb. Of course, this doesn't have to be true.

**50:41** · We can arbitrarily change the ratios between head dimensions and model dimensions. But most models do follow this guideline.

**50:48** · And it turns out to work pretty well. And we can look at a variety of different models, classic and new. I have the latest and greatest coin as well. And you find, yeah, the ratios are roughly around one a model head, notable exception of T5 and even lambda, which is another Google model.

**51:07** · But really, everyone sticks around one. And I think this is an interesting one.

**51:13** · I think the thing about head dimensions that I'll end with here is, I think this is yet another forgiving hyperparameter.

**51:20** · There's a couple ablations that people have done. There's, once again, a pretty wide basin around one that you can get away with.

**51:28** · But that one's maybe not the most critical hyperparameter. I think maybe one of the most critical and interesting ones, I think, conceptually, is this idea of an aspect ratio. And to add an extra point here, when you scale models up or down, the way you usually do that is you fix an aspect ratio, like how wide your model is versus how deep it is.

**51:48** · And then you make the model bigger. So the aspect ratio, in some sense, controls the entire depth to width trade off as you make models bigger. Now, you might wonder how deep should my model be.

**51:59** · If you've been following all this stuff on reasoning and so on, you might think I need a really deep model or really shallow model if I want systems utilization. You might think that there's a lot of variation. And there is a lot of variation, much more so than other hyperparameters. But there's actually a fairly clear sweet spot that most modern models fall into.

**52:20** · You don't really see models go too deep. And you also don't see models go too wide in either direction.

**52:29** · You see, most models have a ratio of about 100 D model over n layers.

**52:34** · So about 100s width for every layer that you have. I mean, this is true for GPT 3, or Llama, or any one of these models. And really, I think the considerations are partly a trade off between expressiveness and hardware. If you have an extremely deep model, they get very, very annoying to deal with, systems-wise. The deeper your model, what is the ways that you have for parallelizing them?

**53:00** · Well, you might have to cut up your layers. If you cut up your layers, we'll talk about this in the systems lecture.

**53:05** · Once you start cutting up your layers depth-wise, you have very serious issues in parallelization.

**53:11** · Pipeline parallel, which is what this is called, is something that most people really, really do not want to deal with.

**53:17** · Whereas width is much easier to parallelize. If you have a really wide model, you can cut that up very easily in your GPUs. Tensor parallel is what it's called. It's much, much simpler to deal with.

**53:29** · And so in some sense, there are systems reasons to go wide. And maybe there's expressiveness reasons to go deep.

**53:35** · And you end up at roughly 100. And I think one of the really interesting things about transformer hyperparameters is, there are a lot of hyperparameters that seem quite important, but they're also fairly forgiving.

**53:49** · And people have converged roughly on the minimum. This is yet another plot from Kaplan, et al, which shows another sweep over hyperparameters for differently sized models. And once again, you see, regardless of the size of your model, roughly speaking, the optimum aspect ratio is fairly similar, and they live at about hundreds, maybe a little bit less, depending on how you want to do the accounting. But really, anywhere near 100 is a pretty safe bet for aspect ratios EK and others did a number of really interesting architecture

**54:25** · variation experiments, in which their general conclusion on this was that, let's look at the top panel here, you have a lot of different kinds of models that you can have in terms of depth to width trade offs.

**54:37** · But as you sweep the depth to width trade offs, you find that really, the only thing that matters, in some sense, is flops. As you increase the flops, the models get better. And that's really controlling the majority of the effects, not necessarily the aspect ratio. And so I think what has really emerged from this is the sense that there's a general forgiving band of hyperparameters that people tend to choose. And then you really worry about primarily, your system's utilization rather than expressiveness concerns, which are hard to reason about. Cool.

**55:11** · And then maybe the last hyperparameter thing that I want to mention is vocabulary sizes.

**55:18** · And this one's interesting to me, because there's a really clear difference between two classes of models.

**55:24** · I think, in the early days of open source model training, there were a lot of monolingual models, whose only goal was to be good on English. And for those models, you had these much smaller vocab sizes, in the 30,000 range. And then post Llama, a lot of people were really interested in multilingual or production systems, these include closed source models like GPT 4.

**55:51** · All of these have much, much larger vocab sizes. And these are roughly in the 100,000 to 200,000 vocab range.

**55:58** · And you see, generally, that Google models have a ton more vocab. Llama derivatives roughly range at about 100,000 tokens.

**56:06** · And then the monolingual models are about 30,000. This is somewhat clear.

**56:12** · The multilingual models really do need much larger vocab to cover the whole space. Generally, the models on the right are also bigger.

**56:18** · There have been scaling law studies, showing that the bigger your model, the larger the vocab it can handle.

**56:23** · And so this is also partially driven by modern scaling trends, where the models on the right are generally bigger.

**56:28** · No ones training large monolingual models anymore.

**56:35** · So yeah. \[INAUDIBLE\] the style stuff.

**56:45** · How to use those vocab sizes? Sorry, the question was, if you have multilingual models or-- Multimodal. Multimodal. Yeah. So I guess, it depends on the way that your tokens are encoded.

**56:58** · But if you're tokenizing your images and things like that, then you need to have many more tokens to account for those.

**57:04** · If you look at various open source releases, they'll have a different image tokenizer with its own vocab, which is quite large.

**57:10** · Yeah. How valid is it to compare, I suppose, this provides for different \[INAUDIBLE\]? How valid is it to compare bits?

**57:23** · Oh, that is a great question. OK. Yeah. That's not a hyperparameter question, but that's a good question. So what is the right way?

**57:30** · So let me step back a moment and put us in the right mindset. So if we think about language modeling, language modeling is a generative modeling task. We are modeling the probability of a sequence. Now, as long as your sequence is fixed, you have adulterated it anyway. And you provide a probability over all strings, that's always valid to compare.

**57:51** · At that level of things is always valid. Now, when you ask the question, is it valid to compare the bits per byte of two arbitrary tokenizers? Really, there's two things at play.

**58:04** · The one thing is, did you touch the sequence at all? If you look at some tokenizers in the past before subword tokenizers, they would drop some tokens or drop some words. That makes the comparisons invalid.

**58:16** · But modern tokenizers are complete. They can model any sequence, so that's not a concern. The other thing that you have to worry about is, are we length normalizing in any way? But for bits per byte, you're always normalizing with the same number, which is the number of bytes. And so this is always a valid comparison. So that's how to think about tokenizer comparisons.

**58:34** · So for example, I think there have been results showing that comparing perplexity for a fixed tokenizers always leads to better downstream performance on downstream tasks, as the same thing improving for \[INAUDIBLE\].

**58:52** · Perplexity and BPP are dual to each other. So yes, if that's what you're asking.

**58:58** · It's only yes and it's only no, because if you're comparing-- you could train compared to perplexity as \[INAUDIBLE\], but you're changing a different thing.

**59:10** · \[INAUDIBLE\] OK. We'll have to talk later, because I'm not sure I understand the question.

**59:15** · But I think that's an interesting set of questions. OK, good. OK. All right.

**59:21** · So we're going through, really, the lowest levels of details of language modeling, which I think really exposes a lot of interesting ideas while we talk through this.

**59:33** · And I think dropout is one of the end regularization. I think it's another very interesting class of ideas, also one that I think is very counterintuitive from your Machine Learning 101 intuition.

**59:44** · So let's go through what I think is the standard argument for regularization.

**59:51** · Well, if I'm doing language modeling, I have a lot of data. I have more data than I can process most of the time.

**59:57** · Unless you're at Google, maybe even then, There is more internet data than there is flops.

**1:00:03** · So I'm probably not even going to see the same data twice.

**1:00:08** · So I'm only going to do a single pass on a corpus. And there's very good reasons and arguments to believe that a single pass of SGD or other optimizers is never really going to memorize my data very much. So this means overfitting is not really a problem, almost ever during compute constrained language modeling. Now, some people even actually only look at training loss, because they believe so strongly that overfitting doesn't happen in single pass SGD. Now, given this, you can sit and think about this.

**1:00:38** · Should I use dropout or weight decay in language model training? You can think about it a bit.

**1:00:47** · One unfortunate thing is that a lot of recent models don't talk about this stuff at all. It's really lower level details than tech reports are willing to expose. But if you look, actually, you find a lot of models do both, especially weight decay actually is a fairly popular intervention even for modern high-performance language models. This is very, very surprising.

**1:01:14** · I mean, some of the dropout things maybe have gone out of favor, but weightdecay actually remains fairly popular. And this is very mystifying. Why is this? And this is one of the reasons why I think deep learning is hard. And this architecture lecture is very strange and hard. It's because these things interact in very strange ways.

**1:01:36** · So there have been papers that have argued and shown nice evidence that weight decay is actually not a regularizer, sometimes. It actually interacts with the optimizer to essentially make optimization better. So if you look at the training versus validation loss across different weight decay settings on language model training for single pass SGD, you don't really see any difference. Weight decay isn't shifting things, so the validation loss is better. There's already no overfitting.

**1:02:08** · Run the x equals y line here. So doesn't control overfitting, but if we look at different levels of weight decay, and not only just different levels of weight decay, we look at weight decay combined with learning rate decay. What we find is that the stronger weight decay runs, these blue dashed lines on the bottom do significantly better, because they start out slow, but they essentially end up converging to a much better minimum later.

**1:02:39** · And this is generally true when we decay learning rate, not necessarily true when we're in constant learning rate, which is maybe somewhat more of where your intuition is coming from. So this is part of why it's very difficult to reason, a priori or from scratch, the behavior of all these different choices and why I think Percy and I have designed this class, so that you interact with stuff, because you might come upon this thing where basically, weight decay is actually an optimization intervention and not necessarily a regularization intervention,

**1:03:12** · which is what you would expect here. So always keep that in mind that these kinds of unexpected effects can really start to kick in for these kinds of settings.

**1:03:24** · Cool. All right. So to put everything together for hyperparameters, there's actually, for a lot of the maybe more hairy looking hyperparameters, actually just fairly standard choices that have worked well for everybody. Factor of 4 rule of thumb, keep your head dim and your number of heads equal to the model dimension.

**1:03:45** · Pick an aspect ratio, roughly around 100. And if you ask about regularization, you want to maybe try a couple of things, because regularization actually does interact with optimizers in ways that are quite counterintuitive.

**1:03:58** · So this is the thing that some people still do, even though you don't need the regularization at all.

**1:04:04** · Actually, maybe I'll stop here, in case. Yeah. Are there any significant architectural differences maybe for diffusion model? Diffusions. That, I have not looked into enough, to be honest.

**1:04:16** · There aren't that many people training. Big diffusion is one issue. And many of the models that have been trained are retrofitted.

**1:04:23** · So I think the architectures are actually the same as a Llama-like model. But if you're asking the question, what's the optimal architecture if you were to train from scratch, I don't know what that is, actually, off the top of my head.

**1:04:34** · Yeah. Do you have any explanation for why regularization mostly affects \[INAUDIBLE\]?

**1:04:39** · Well, I guess, it's not that regularization, in general, affects optimization. I don't think people do dropout anymore, because it doesn't really interact well with optimization.

**1:04:47** · But for example, weight decay, which is shrinkage to 0, that might allow you to use a higher learning rate, or it might allow you to decay faster. There are lots of ways in which all of these terms are interrelated.

**1:04:59** · Cool. OK. Now, I've talked a lot about how to design expressive models by looking at all these other models that have been trained. One of the things that I'll highlight now is, over the last few years, a really big emphasis has not been on performance alone, it has actually been on stability. And this becomes an increasingly important concern as your models get more and more expensive to train.

**1:05:26** · We've seen that a lot of these choices are forgiving. Everyone's doing similar stuff.

**1:05:31** · And so you can mess with these, but you're not going to get a big performance difference. That's fine. But if your model suddenly blows up some part into training, you get these horrible-looking spikes all over the place, you might end up with a model that is actually not very good quality. Or it might be unrecoverable. You might have spent millions of dollars in training.

**1:05:52** · And you get to a point where the model is no longer able to be trained any further. That would be a horrible thing to happen if you have a lot of compute that you want to spend. So you don't want to train models that look like this blue curve with spikes everywhere and these big gradient norms happening. So what do we do to fix these stability issues?

**1:06:10** · I mean, this is really, I would say, a core, core issue. And if you have stability issues in language models, or in general, neural networks, there's a few usual suspects that you've got to start looking at.

**1:06:24** · One of them is the softmaxes. And the softmax has two things that are both really bad for stability-- one of them is an exponential.

**1:06:32** · You can see how that blows up very quickly. You also divide two numbers. And that's also a potentially very dangerous operation.

**1:06:39** · So softmax is one place where you got to be extra, extra careful. And where are the softmaxes in a language model?

**1:06:46** · Well, there's two of them. There's one on the output side, when we output our probability distribution.

**1:06:51** · And then in attention, when we normalize the attention, there's going to be another softmax. So we can think of both of those as really danger zones for our model, especially the attention.

**1:07:06** · Let's start with thinking about the output softmax. The output softmax can blow up on us.

**1:07:12** · And one of the things that we can do is, we can try to control the normalizer problem.

**1:07:19** · So let's think about the softmax calculation. We want to compute a log probability to compute the loss.

**1:07:25** · Now, what is the log probability? Well, it's the output of your model u. And then you've got this log normalizer.

**1:07:32** · This u is well-behaved, because in some sense, this is the output of your model. This is just the output of your residual stream with all the things that are added in. So if u is well-behaved, then the first term is well-behaved, if the model is being OK.

**1:07:45** · Now, the second term, this log z, this might not be so OK. If z is really big or really small, even if the output of your model is somewhat well-behaved, it could blow up. And what is z? Well, it's an exponential.

**1:07:57** · So it could potentially blow up very quickly on you. Or if this is 0, it could also blow up on you. So both of those directions are very, very bad.

**1:08:04** · Now, we would ideally like our z to be somewhere near 1 or log z to be somewhere near 0. What can we do? Well, one of the things that you notice, if you thought about the action of the softmax, is, this whole thing is overparametrized.

**1:08:23** · I could push things in and out, so if I add a constant to you, I can manipulate the z's without really affecting the output of the softmax. It can cancel out between the normalizer and the output of my model.

**1:08:35** · So because of that property, one thing that I could do is I could add a regularizer. This is from Jacob Devlin's paper, 2024-- sorry, 2014, in which he adds this squared log z term. And what this is doing is it's just penalizing how far away your log z is from 0. And if log z is near 0, that's nice, because this whole expression is numerically stable.

**1:09:01** · This is called the z loss trick. It's been used by a number of papers. Jacob Devlin and others initially pioneered this back in 2014. And then it's become popular again through a number of open source models. Baichuan, I think, was the first open source model to do it, but then DCLM and OLMo, and others have been using this trick to stabilize their output softmaxes.

**1:09:23** · So this is a surprisingly effective thing. So let's say, we've handled the instability issues on the output softmax. Now, we have to turn our attention towards the other potential problem, which is attention.

**1:09:37** · And this is a place where lots of degeneracies happen. Lots of techniques have been developed to control the instability that attention operations generate.

**1:09:48** · And really, the high level thing that I'll say is, if you have instability, if you can throw a layer norm in there, somehow, it might control it. And that's really, in some sense, the design philosophy behind this idea called the QK norm.

**1:10:04** · So what you do is remember that we have our Qs and Ks that are going to be multiplied together, and then they're going to go into the softmax. So in the standard attention operation, I'm going to layer norm as a prelayer norm, multiply with a QKV. And I'm going to get my Qs and Ks. Those will get multiplied by a matrix multiply, I'll softmax them, and I'll multiply that with a V to get the weighted average. And then I'll output whatever comes after right.

**1:10:29** · So this is our usual attention. Now, what happens if we just throw in a LayerNorm before we multiply the Qs and Ks? If we do that, then that the inputs to this matrix multiply and therefore, the inputs to the softmax, roughly have the same scale. They're always going to have a scale of roughly one, because we've used the RMSE norm to divide the size of those Qs and Ks.

**1:10:55** · If we do that, then we're going to keep this softmax operation stable.

**1:11:00** · Tons of different models do this. It's originally from the multimodal world.

**1:11:06** · Some folks who were making multimodal models initially discovered QK norm, Idefcs, and Chameleon really use this and proved it out. And then a number of other open source language models realize that the same tricks are entirely applicable to stabilizing attention for language models.

**1:11:26** · And I think this is now very, very standard. QK norm is actually a very standard intervention that most of the large models now introduce.

**1:11:34** · It doesn't seem to affect performance from lots of different training runs, but it does definitely prevent the kinds of attention degeneracies. And really, the way that I've seen this is, we have layer norms, initially in the prenorm. Now, we add them after the nonlinearities in each block.

**1:11:54** · And now, we're throwing them in both the Qs and the Ks. And really, I think this is getting at the stabilization tricks that people apply to this world. Now, the final set of things that I'll talk about as a stability intervention, and frankly, this one is not as popular and more of a Google-specific trick that I've seen, but logit soft-capping is a much harder intervention that some people apply.

**1:12:21** · So this one, in QK norm, what we're doing is we are controlling the inputs to the softmax and hoping that the outputs are well-behaved. If we really, really want to enforce well-behaved outputs, what we can do is we can take the logits, the things that go straight into the softmax, and we can just cap them off, so they can never be too large or too small. This is almost a hard constraint. It's called a soft cap, of course, but a Tanh is bounded at some value. And so this is in the Gemma models.

**1:12:54** · I think Gemma 2, 3, and 4 all use the logit soft cap trick.

**1:13:00** · And what they do is they take all of their logits for the attention layers, and then they soft cap them at some value.

**1:13:08** · Some NVIDIA folks have done actually quite nice work doing systematic comparisons of these stability interventions.

**1:13:14** · And what they find is, if you start with a baseline model, you can do all sorts of different interventions.

**1:13:21** · And QK norm is here. And it does slightly better due to the fact that you can crank up the learning rate a little bit.

**1:13:28** · But if you do soft-capping alone, you actually end up losing performance. So there is a quality degradation that happens.

**1:13:34** · This is a very strong intervention. You can never express very confident signals in your softmax beyond a certain point.

**1:13:41** · So it does have some negative consequences, but this is a very safe way of stabilizing the outputs of your attention-- or sorry, the inputs to your attention, the logits that go into the softmax.

**1:13:52** · So that's the end of the stability components. I can pause for a moment here.

**1:13:59** · And I'll talk about various attention things after that.

**1:14:06** · Cool. OK. All right. So the last thing I want to talk about today is various interventions that you can make to your attention. And as I was saying at the beginning of this lecture, I'm only going to talk about all the things that you can do to dance all by all attention today.

**1:14:23** · So if you were interested in hearing about state space models or linear time attention, sadly, today is not the day for you. The things that I do want to talk about, which are really commonly implemented attention interventions today, are group query attention, which really saves inference costs by reducing the number of heads and sparse or sliding window attention, which really originally came from the GPT 3-ish family, but have now really been adopted widely by most models that are looking to do long context, unless they're doing exotic SSM stuff. So I'll start with group query attention, or GQA, or MQA.

**1:15:06** · I'm going to first set up the need for these kinds of things. And then you'll hopefully see what the trick is and why it's fairly natural. So for the moment, we've been talking about training, and modeling, and all these things, but let's take a pause. And now, let's think about deployment. You've trained this very big model.

**1:15:26** · And now, you need to serve it to lots of users. And you're going to pay a cost for serving. And you're going to have to, in abstract sense, pay for two different resources. You're going to have to pay for your flops, the computation that you're performing.

**1:15:40** · But you also have to pay for another thing. You have to pay for your memory accesses. Because the memory accesses are also going to impact your systems characteristics, your latency, your utilization. So you want both of these things to be small.

**1:15:53** · Now, let's think about what happens during training or alternatively, prefill, when you're looking at your prompt, where someone gives you the stuff. In this case, the total arithmetic operations you have is order of magnitude, batch size, sequence length, hidden dim squared.

**1:16:10** · That's roughly the size of things that you get. And of course, we're doing quadratic attention, so we've got d squared.

**1:16:17** · We've got total memory accesses. What is our memory access that we have here?

**1:16:22** · We have batch times sequence length times hidden dim plus the cost of the softmax, which has an n squared component. And then we've got a d squared component for the projections.

**1:16:34** · So the arithmetic intensity here is pretty good. It's going to be 1 over k. This is the number of heads.

**1:16:39** · So you need to have, sorry, head dims. So your head dims need to be big enough that you're multiplying some reasonably sized matrices.

**1:16:46** · And you've got a 1 over bn, so your sequences need to be long enough or your batch sizes need to be big enough.

**1:16:51** · As long as both of these are true, your GPUs are going to be fully utilized. Great. You're using all of your resources.

**1:16:59** · Now, we have finished training. And now, we're serving our users. How do we serve our users?

**1:17:05** · We're going to generate tokens and send it to them. Now, if we're doing that, I can't parallelize the generation process. What I'm going to do is, I'm going to generate a token. I'm going to condition on it. I'm going to generate the next token. And I'm going to repeat this process one by one.

**1:17:18** · This is just the curse of autoregressive language modeling. We have to do this.

**1:17:23** · In order to do this, the efficient way to do it is to maintain all of the passkeys and queries that I've had in what's called a KV cache. So the KV cache is going to maintain this matrix of Q dot K over the past. And then whenever I need to compute something new, I can reuse the submatrices that I've already had from the past.

**1:17:45** · And I only really need to compute the new query key interactions that I need to fill out the rest of this matrix. So every submatrix I've computed before, I can keep.

**1:17:57** · I only need to compute my new one. So this saves a lot on compute. That's great. But the issue here is, now, our arithmetic intensity is not so good. As you might intuit, this KV cache approach is going to be reading parameters all the time.

**1:18:16** · Each time I have a new step, I'm going to have to read in my parameters. I'm going to have to take these dot products.

**1:18:21** · And I'm going to do this once every step. And so now, what do I have? Well, you my total memory-- oh, sorry.

**1:18:28** · My total arithmetic operations are the same. I'm multiplying the same matrices still, just incrementally rather than all at once.

**1:18:35** · But because I'm doing this incrementally, now, I have a memory access pattern of batch by sequence, squared by hidden dim plus sequence by hidden dim squared.

**1:18:47** · And the second term is not so pleasant. It used to be that it was just d squared, but now, we've got n times d squared. And if we compute the arithmetic intensity, which is the ratio of these two guys, now, we have n over d plus 1 over b. So now, what we need is large batches plus short sequence length or we need really big model dimensions.

**1:19:11** · So if we want to serve a small model efficiently, this is not so good.

**1:19:17** · This is really difficult to deal with. This n over d term, this first term over here, which is sequence length over hidden dim, is very difficult to reduce if we're doing this incremental computation. This is just a hard thing to deal with.

**1:19:32** · So this leads to this idea of MQA, or Multi-Query Attention. Normally, you have multiple heads in your attention operation. And you're going to have different keys, different values, and different queries.

**1:19:44** · That's normally how you do things. But one thing that we could do is, maybe we can keep the Ks and the Vs the same across all the heads.

**1:19:52** · And the only thing that's different across the heads are the queries. If we do this, then this drastically removes the amount of items that need to be moved in and out of memory. Because the KV cache are now significantly smaller.

**1:20:07** · These are all shared across all the heads. This significantly reduces the total memory access as well as the arithmetic intensity. And the key term that we were talking about here, we had the n over d term.

**1:20:20** · Now, we have h multiplying this. And so this h term allows us to significantly reduce the-- sorry, increase the arithmetic intensity if we have a lot of heads. This is a significant gain over what we had before.

**1:20:36** · So this gets us significant efficiency improvements, but the issue with MQA, this is on the right here, you have one value and one key for all these queries, you do, in fact, lose significant expressive power if you do this. And so there's this trade off between systems efficiency and expressiveness.

**1:20:57** · And you might wonder, is there a sweet spot in which we can avoid trading off quite significantly expressive power and computation? And that's where GQA, or Grouped Query Attention, comes in.

**1:21:10** · The original transformer is multihead. We have queries and keys for each head. In multi-query, we have one key and value for each.

**1:21:18** · For all the heads in grouped query, we reduce the amount of keys and values, but we keep the number of queries the same.

**1:21:24** · So we now have this ratio that we can play with, which is the number key heads or the number of value heads while keeping the total number of heads much larger than that. So this allows us to very simply control the trade offs between expressiveness and inference efficiency.

**1:21:40** · There are other tricks from DeepSeek V2, multihead latent attention that I'll mention briefly next time, which have a different kind of factorization structure and a different set of trade offs. But really, the nice thing about GQA is that, in practice, the trade off is quite favorable. So if you have multihead, your performance, this was in, I think, T5 days, if I remember right. This is your downstream model performance.

**1:22:08** · This is your time per sample. You want to reduce this as much as possible. With multihead attention, you have best performance but very high cost. With MQA, you have lower cost but much lower performance.

**1:22:22** · Similarly, if you make your model smaller to try to hit your performance targets, you get much worse performance, GQA really does get the best of both worlds. Very low inference cost, nearly the same performance as your full multihead. And you see this GQA group structure, where if you have a small reduction in the number of heads, you basically have most of the gains in your performance, which allows you to keep most of the expressive power while getting significant inference improvements.

**1:22:54** · And Percy will talk a bunch more about the inference mechanics later, but this should give you a flavor why models today, almost all, adopt this GQA structure, because it gives you a lot of this inference cost, which is really critical, without very much of a expressiveness hit.

**1:23:14** · Cool. Any questions for GQA or KV cache? Yeah. \[INAUDIBLE\] given that you have so many rules of thumbs for what hyperparameters are \[INAUDIBLE\] to what extent are you still searching over hyperparameters versus exploiting these rules of thumb \[INAUDIBLE\]?

**1:23:33** · I think it's a mix of both. I think every model training run has some thesis about what can be varied.

**1:23:40** · And so you see this in a lot of the reports, where I think the hyperparameters are often not where people are touching too much, but you see architecture changes one at a time in a lot of these reports. But it's very rare to go and change everything up. I think Google is one of the only orgs that seems to really spice things up in a significant way. The Gemma series has done some pretty interesting things.

**1:24:02** · The most recent Gemma 4 release, for example, now has an individual embedding for every layer in a way to control the trade offs between memory use and flops. It's a very interesting set of things that they've done.

**1:24:14** · Oh, yeah. Back there. \[INAUDIBLE\] experiments with dynamically altering some of these parameters during training.

**1:24:22** · During training. Let me think. Weight decay, yes.

**1:24:27** · Weight decay, people change in concert with learning rate. That is actually a heuristic that people do.

**1:24:34** · That works very well. Other than that, I don't know if there's a lot of different hypers that people change during training, especially because the architecture ones just make training incompatible. So you can't really change them while you're training.

**1:24:49** · So I think weight decay is probably the one that I can think of. The others are usually fixed.

**1:24:54** · Yeah. So \[INAUDIBLE\] MQA, it's not just inference time, it's a \[INAUDIBLE\]. That's right. Yeah. You train with a certain number of repeats.

**1:25:07** · Cool. The last thing I'll talk about is sliding window attention, which is a really old idea.

**1:25:14** · GPT 3 used, actually, this-- if you read the paper, they'll say, we alternate between full attention, where every position can attend to everyone in the past, and a banded matrix style attention, where you can attend to everyone within a fixed window.

**1:25:30** · And OpenAI has some early work on these different kinds of attention patterns that you can use.

**1:25:37** · But actually, this has become really, really popular over the past year. This idea of alternating the big, full attention and a more local attention actually hits a sweet spot for how to manage long context performance while not paying too much for inference. I think the more recent revival in open models,

**1:26:01** · I would maybe say, Cohere Command A was the first one I saw do it, where they had this structure, where every four layers, they would have a full attention that attended to everything. The three layers in between would use a sliding window attention that would only be able to look at local structure. And of course, as you go up, sorry, in this case, down, because they ordered the diagram the other way, as you go down these blocks, you're aggregating local information into global ones. So the local attentions at the end can, of course, access more global information,

**1:26:32** · but this allows you to manage the cost of having a really long context without having to go for something like a state space model or a more exotic intervention.

**1:26:44** · And that's worked quite well. There's also some innovation, where people change the embedding format for the long range information, where they get rid of things like RoPE. So you have no position embeddings at all. So you're really looking, almost, at bags, where the short range information still gets position information. So people do all sorts kinds of interventions involving these both embeddings and alternating local and global structure.

**1:27:13** · I'll say that this is attention and in general, how to manage the trade off between long context cost and performance is still an active area of investigation. It's a place, where the most architecture work and changes are still being done. We see, essentially, a bunch of other models adopt this idea-- Llama 4, most recently, Gemma 4, OLMo 3, they all do this combination of sliding window attention and full attention, in their case, using full RoPE instead of NoPE as the embedding.

**1:27:45** · So as I said, this is becoming really, really popular. Qwen3.5, which I've put on the right, they're actually a little bit different, because they alternate a state space model called gated delta net and one full attention for every four layers.

**1:28:04** · So it's the same alternating structure, but they're using a different cheap layer. In their case, they're using a state space model, I'll explain what that is next lecture, instead of a sliding window local attention.

**1:28:15** · But you see, this is like, I think, a new theme over the past year, where open models are really trying to grapple with long context performance. And the way to do that, at least, so far, is to have these hybrid models that aren't just global attention, aren't just cheap attention, they're some of mix in between. And that seems to have worked very well so far in a lot of these models. OK. Cool. So as I was trying to emphasize, when

**1:28:41** · you look across all of these models, you start to see a lot of patterns, and hopefully, a sense of general understanding about what things you can do and what things are good defaults. We also see a lot of differences in how we handle context and how we handle position embeddings. Even tokenization, there's some differences. So there are differences across these models, but there's also commonalities that hopefully, now, give you some intuition as you go out, and do your assignments, and mess with the leaderboard and so on.

**1:29:08** · Thanks.
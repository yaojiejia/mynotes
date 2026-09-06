---
title: "Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 2: PyTorch (einops)"
source: "https://www.youtube.com/watch?v=kuYAsz7zspQ&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=2"
author:
  - "[[Stanford Online]]"
published:
created: 2026-09-05
description: "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."
tags:
  - "clippings"
---
![](https://www.youtube.com/watch?v=kuYAsz7zspQ)

## Transcript

**0:05** · I hope everyone is staying dry.

**0:08** · I'm not.

**0:09** · So as I mentioned last time, the Marin project had a 1e23 FLOPs which was running, and it finished.

**0:16** · And it actually matched the forecast.

**0:19** · So remember we were running this.

**0:21** · Each of these curves is essentially IsoFLOPs curve, which is a bunch of smaller model runs.

**0:28** · And you try to find the compute optimal point.

**0:30** · You fit a scaling law.

**0:32** · And this was the point where we predicted the loss, and we ran the model and it got loss within 0.05.

**0:42** · So I thought that was pretty cool.

**0:44** · And if you extrapolate out to GPT-5 level performance, this is the loss you get.

**0:50** · Of course, your mileage might vary depending on how these scaling loss are.

**0:56** · OK, I just wanted to share that news.

**1:00** · So last lecture I gave an overview of the entire class.

**1:05** · And we are talking about tokenization, which is going to be on the first assignment.

**1:10** · Today, I want to talk about resource accounting, which is going to be more on the systems side of things.

**1:20** · So to recall, the main thing we're trying to do is train the best model we can given a finite set of resources, which could be compute, memory, sometimes data.

**1:34** · But that's not really going to be a limiting factor for us in this class.

**1:39** · And our goal is to simply to maximize the computational efficiency of our training.

**1:46** · So before you can optimize the computational efficiency, we need to understand the efficiency of a given computation.

**1:53** · And for that, we need to understand the compute and memory characteristics.

**1:58** · Just to give you a taste of the type of questions you will hopefully be able to answer by the end of the class, so here's a question.

**2:05** · How long would it take to train a 70 billion parameter model on 15 trillion tokens on 1024-- actually, this should be H100.

**2:16** · So how do you answer that?

**2:17** · Well, there's a formula, which we'll talk about how you can get the number of FLOPs to be 6 times the number of parameters times the number of tokens.

**2:26** · We can look up the spec sheet to see how fast the H100 is.

**2:32** · We have this thing which called MFU, which we'll talk about 0.5.

**2:36** · Then you can estimate the number of FLOPs that you need that the hardware gives you per day.

**2:44** · And then you can compute the number of days.

**2:46** · So the number of days is 143.

**2:50** · Here's another question.

**2:51** · What's the largest model you can train on H100s using AdamW?

**2:59** · So you can look at, well, H100s have 80 gigabytes of each VM memory, the number of bytes per parameter, which is 2 plus 2 plus 4 plus 4.

**3:17** · We'll explain where that comes from.

**3:19** · And then the number of parameters is that you can get is going to be about 53 billion.

**3:27** · So there's some caveats here that we don't count the activations which depends on batch size and the sequence length.

**3:35** · So this is all very rough back-of-the-envelope calculations.

**3:38** · But hopefully by the end of this class you'll understand where these come from.

**3:43** · And the point is not to precisely calculate every single thing, but just get the rough shape of things.

**3:54** · OK.

**3:54** · So last time I talked about knowledge and what you can take away from this class-- mechanics, which are how things work.

**4:03** · So today, that would be pretty straightforward.

**4:05** · The mechanics are just how PyTorch work, how tensors work.

**4:09** · It should be fairly-- there's no magic here.

**4:13** · The mindset I want to impart on you is that resource accounting is going to be very crucial.

**4:20** · And I want everyone to get in the habit whenever you write this line of code, think about the performance characteristics.

**4:30** · And then finally intuitions.

**4:32** · Here we're just going to get a sense of the resources, how they're spent.

**4:37** · There's going to be no ML magic today.

**4:39** · I'll leave that to Tatsu for the next lecture.

**4:42** · OK.

**4:43** · So let's get into things.

**4:46** · So let's start bottom up and start building up.

**4:49** · So what is at the bottom?

**4:52** · At the bottom are tensor.

**4:53** · So tensors are the building block of storing everything.

**4:56** · If you have parameters gradients, optimizer states, data activation, everything essentially is a tensor.

**5:04** · So for example, you can take a look at the DeepSeek 3.2 model.

**5:10** · And you see that the model itself is a bunch of different tensors.

**5:14** · Each tensor has some shapes and also some precision, which I'll talk about later.

**5:20** · And so as you know, tensor is subsume vectors, matrices, and can generalize to any number of entities.

**5:33** · OK.

**5:33** · So let's talk about how much tensors take to store.

**5:42** · So it depends on the type of tensor.

**5:46** · So in general, we're going to be dealing with tensors that store floating point, but tensors can also store integers and other types.

**5:56** · So for floating point, typically whenever you talk about float, I think the standard people refer to as float what is called float32.

**6:08** · So a float32, if you break it down, has 32 bits.

**6:14** · One of the bits is a sign, 8 bits are the exponent, which gives you dynamic range.

**6:19** · And the rest is the mantissa or the fraction, which gives you variation.

**6:26** · This is also known as fp32 or single precision.

**6:34** · And the term single precision comes from the fact that back in the day when you were doing scientific computing, float32 was like a baseline.

**6:42** · It was just like you would expect if someone gave you a float, you would expect it to be at least single precision.

**6:47** · And if you want more precision, then you can get double precision as that's float64.

**6:56** · But in deep learning, we're going the other way, because even 32 is a lot.

**7:02** · And the types of computations that we want to do don't demand the high precision that some kind of numeric simulations do.

**7:10** · OK.

**7:11** · So before we get to other types, let's just look at float32.

**7:15** · So let's construct a 4 by 8 matrix.

**7:18** · By default, the type of tensor you create is float32.

**7:23** · So if you want something else, you should declare it.

**7:26** · And the memory usage is just the number of elements times the element size here, which is 4 bytes for a 32-bit number.

**7:36** · And that's going to be 128 bytes.

**7:41** · So just to give you a kind of perspective, so in GPT-3, which is a fairly old model one of the matrix in the feedforward layer is about 2.3 gigabytes.

**7:56** · So these tensors can get quite big.

**7:59** · And this is not even the biggest one that one can imagine.

**8:04** · OK.

**8:06** · So since we're interested in efficiency, we want to generally reduce the amount of storage.

**8:14** · And we'll see that as you reduce the precision, you actually save memory.

**8:20** · And you also save time because operating on 16 bits is going to be faster, let's say, twice as fast, but not always.

**8:31** · It depends.

**8:34** · And then memory, by reducing memory, we'll see later that actually reducing memory can save time as well, which is maybe less obvious.

**8:43** · But it will hopefully become clear.

**8:46** · So the obvious thing is you say, OK, let's take away half the bits.

**8:51** · Now you have float16.

**8:52** · So float16 says you have a sign.

**8:54** · You have only 5 bits of exponent, and then the rest is the mantissa.

**9:00** · So float16 is good, except for its dynamic range is poor.

**9:10** · So even if you have, let's say, try to construct 1e minus 8 tensor, then that is actually just 0.

**9:19** · So you can't really represent very big numbers, and you can't represent very small numbers.

**9:25** · And the reason is that this exponent it's only 5 bits of exponent compared to 8.

**9:35** · So if you train with fp16, which people did back in the day, you will get instability.

**9:46** · You will get underflow.

**9:47** · You get overflow.

**9:48** · You'll get NaNs.

**9:49** · It's pretty challenging.

**9:53** · So bfloat16 was invented.

**9:58** · So this was developed in actually 2018 to address this issue.

**10:03** · And the observation was that well, let's not compromise on the number of bits.

**10:10** · The number of bits is going to be the same as fp16.

**10:13** · But we're going to shift some of the bits from the mantissa to the exponent.

**10:18** · So that means it has more dynamic range than float16.

**10:26** · And it actually has the same dynamic range as float32.

**10:30** · But of course, the resolution is worse because there's no free lunch here.

**10:35** · But it turns out that in a lot of deep learning applications, this is well worth the trade-off.

**10:40** · You want the dynamic range to not overflow and underflow.

**10:44** · And because things are kind of sloppy and stochastic anyway, you don't need that much resolution.

**10:54** · So let me actually skip over this part.

**11:00** · So OK.

**11:01** · So to summarize, what are the implications for training?

**11:05** · So you can absolutely train with float32.

**11:08** · And if you're training a small model, you probably just-- and you don't want to worry about it, just float32 is fine.

**11:15** · But it requires 4 bytes of-- sorry, 4 bytes of memory per float.

**11:25** · And that can take up a lot of memory.

**11:28** · And if you train with float16, then that's going to be too risky.

**11:35** · So bf16 is the sweet spot.

**11:42** · Even bf16 can be risky as well.

**11:45** · Maybe \[INAUDIBLE\] a little bit.

**11:49** · One thing that people have-- has become common practice is to use mixed precision training.

**11:57** · And the actually let me-- OK, let me compile this again.

**12:05** · I think these are stale.

**12:06** · So mixed precision training is where some of the computations use some precision, and for other computations use other precision.

**12:18** · So in general, as a general rule, bf16 is what you would use for parameters, activations, and gradients.

**12:29** · And for optimizer states, you would use fp32.

**12:32** · And we'll discuss a little bit more about that later.

**12:36** · And to do invoke mixed precision training, so PyTorch has an AMP library.

**12:40** · We're not going to talk too much about this, but you basically wrap your code, and PyTorch has-- the library takes care of it automatically for you in the sense that it tries to cast things into a bf16 when it's safe.

**12:56** · So, for example, MatMuls are generally safe.

**12:58** · But if you try to do exponentiation, then it will try to leave things as fp32.

**13:06** · OK.

**13:06** · So we could do bf16 I think is probably where this class will end.

**13:11** · But if you're feeling very adventurous, you can go farther.

**13:15** · So fp8, this was introduced four years ago.

**13:19** · It actually has been standardized.

**13:22** · So if you look at fp8, there's actually two versions because depending on if you need more dynamic range or more resolution.

**13:32** · And there's two versions of them.

**13:37** · And so we're not going to talk about that.

**13:38** · But NVIDIA has some-- the transformer engine supports fp8.

**13:43** · And most recently, you can actually go down to fp4.

**13:47** · So last year, NVIDIA developed nvfp4, and there's only 4 bits per value.

**13:53** · So if you just to-- so we're on the same page.

**13:55** · 4 bits is not a lot.

**13:56** · I can write all the values down on a single line here between minus 6 and 6.

**14:03** · So that's not very much precision.

**14:05** · Now there's a little bit of a cheat here.

**14:07** · Because if you just naively only use these values, you're not going to be able to train very well.

**14:14** · So what actually this means is that every value you can have this 4 bits of freedom, but there are blocks.

**14:23** · And each block can be scaled up and down accordingly.

**14:26** · So you can actually represent more values but you can't represent the full dynamic range for every single value.

**14:35** · And there is a model released actually this year, the Nemotron 3 Super, which was trained in fp4, which I think is pretty cool.

**14:45** · OK.

**14:46** · So some of this it's just good to know.

**14:50** · And some of this you can't even touch.

**14:52** · It's not like you create a tensor and you call it fp-- you can make it fp4.

**14:58** · A lot of this is done under the hood by NVIDIA's software stack.

**15:05** · OK.

**15:05** · This was a bit of a long digression, but it's maybe helpful to appreciate that the intricacies of precision here.

**15:14** · Any questions before I move on?

**15:17** · Yeah.

**15:18** · Sorry.

**15:19** · It was a question like, when you have a block, you're scaling all those same things in tensor by that block.

**15:25** · So then the ratios of all that will still be in fp4.

**15:30** · Yeah.

**15:30** · So the question is to just-- \[INAUDIBLE\] maximum value is to have more specificity.

**15:36** · Yeah, so the question is, OK, just to explain the block a bit more.

**15:39** · So you have, let's say a block.

**15:41** · Within that block, you can vary up within the 4 bits.

**15:47** · And in addition, all those values can be scaled up and down according to how many bits you're scaling factor is.

**15:54** · So if you look at an individual value, you actually get more than 4 bits of dynamic range.

**16:01** · But it's just like you can't have this value be way over here and the neighboring value way down here.

**16:08** · One question about, what about 1 bits \[INAUDIBLE\] affect 1 bit \[INAUDIBLE\]?

**16:16** · Yeah, so there's a lot of training obviously.

**16:19** · So maybe this is a good point.

**16:21** · So the question is about 1 bit because you can't really go lower than that.

**16:26** · So there's a difference between training and inference.

**16:30** · So a lot of the low bit stuff, if we talk about-- I think we'll talk about quantization later, is that you train a model on and maybe even bf16.

**16:41** · And then you quantize into, let's say, 1 or 2 bits.

**16:46** · And that is much easier than training a 1-bit language model, which I don't think anyone has done.

**16:54** · Maybe it's possible, but I don't think anyone has trained anything credible there.

**16:58** · OK.

**17:00** · Let us go on then.

**17:02** · So we talked about tensors.

**17:04** · And the memory calculus is pretty simple, just the number of elements times however much memory each element takes up.

**17:12** · So by default, the tensors you create in PyTorch are going to be on CPU.

**17:20** · And of course, you have to-- if you want things to go fast, you want to move them to GPU.

**17:27** · And actually, so there's a bit of a-- I have a slight issue with the slides that they were executed on my laptop, which means that I don't have GPU.

**17:36** · So some of the code I'll just show but not execute.

**17:43** · OK.

**17:44** · Let's see.

**17:45** · This is maybe not that interesting, but I think everyone knows how to move tensors to GPU.

**17:52** · But just remember to do that.

**17:53** · Otherwise you won't get your speed-ups.

**17:58** · OK, so we talked about memory of tensors, which is very straightforward.

**18:02** · Now let's talk about computing with tensors.

**18:05** · So before talking about FLOPs accounting for the tensor operations, let's take a little bit of a digression to talk about einops.

**18:15** · So how many of you are familiar used einops before?

**18:19** · OK.

**18:20** · So maybe 2/3.

**18:21** · OK.

**18:22** · Good.

**18:22** · So the motivation behind einops, for those of you who might not be indoctrinated is, it's very easy to mess up.

**18:32** · Or I find it very confusing to look at code such as this.

**18:38** · And you have x and y.

**18:41** · And then you have transpose minus 2 and minus 1.

**18:44** · And you're trying to figure out what minus 2 and minus 1 is.

**18:50** · And so this is maybe the motivation for using variable and names rather than indices.

**19:02** · So einops is a library for manipulating tensors where the dimensions are named.

**19:06** · And this is inspired by Einstein summation notation.

**19:10** · There's a nice tutorial which you can go through.

**19:12** · I'm just going to cover some of the basics.

**19:17** · So basically, the way to think about einsum is a generalized matrix multiplication with good bookkeeping.

**19:25** · So here's an example.

**19:26** · So I have a matrix 3 by 4.

**19:31** · I have a 4 by 3 matrix.

**19:33** · And if I do the MatMul, this is actually pretty nice.

**19:39** · It's pretty easy to understand.

**19:41** · In einops, basically you say x has two dimensions, the row and the column, which I'm going to name seq1 and hidden.

**19:50** · I'm going to have y.

**19:51** · That's a matrix which has also rows and columns, which I'm going to name hidden and seq2.

**19:58** · And I'm going to produce tensor or here a matrix, where the dimensions are indexed by seq1 and seq2.

**20:08** · And anything that is not mentioned here, the hidden gets summed out.

**20:14** · So the way this works is I'm going to sum over-- enumerate over all possible values of all the variables that occur here, which are seq1's hidden and seq2.

**20:28** · And I'm going to basically index into x, index into y, multiply them and dump them, accumulate them into the result z sub seq1 and seq2.

**20:43** · All right.

**20:43** · So you say this is-- come on.

**20:47** · This is much easier than this, Right OK, let's try a more complicated example.

**20:53** · So here's the example I showed before.

**20:56** · Now we have a tensor.

**20:58** · It's 2 by 3 by 4, another tensor 2 by 3 by 4.

**21:03** · And if I were doing things the old way, basically what I'm doing is transposing these last two.

**21:12** · And then because at implicitly batches the dimensions that are not the MatMul dimensions, then you get the answer.

**21:24** · So you have to reason about this a bit.

**21:26** · Einops makes this very clear.

**21:29** · It says there's a batch dimension.

**21:31** · There's seq1 hidden, seq2 hidden.

**21:34** · And I'm just going to produce batch seq1 and seq2.

**21:38** · And notice that there's no transpose because I just-- in some sense I've done the transpose by the naming.

**21:45** · If I had hidden and seq2, then it would be no transpose.

**21:50** · I always get confused by transposes.

**21:52** · And the fact I don't have to think about transposing it makes me happy.

**21:57** · OK.

**21:59** · So if you want to get fancy, you can say, well, batch-- I'm just going to replace with dot, dot, dot.

**22:07** · And this means that if I had, let's say, rank 10 tensor with eight different batching dimensions, I can just write dot dot, dot without enumerating all of them.

**22:20** · And this comes up in language modeling because you might have a batch dimension.

**22:25** · You might have a sequence dimension.

**22:26** · You might have a head dimension.

**22:28** · And you're trying to do this matrix operation for all of them.

**22:31** · And you might not want to have to just worry about this.

**22:36** · And the nice thing is that you can write modular code where you can write this dot, dot, dot without worrying about even the shape of the tensor that comes in.

**22:49** · All right.

**22:50** · So that's eisum which is in the einops library.

**22:55** · There's a reduce.

**22:58** · So this is a generalization of sum, mean, max, and min.

**23:03** · So for example, if you have, let's say, this tensor and you want to sum according to dimension of dim minus 1, which means sum along the last dimension here.

**23:19** · So again, I don't like this notation.

**23:23** · But what you can do is call reduce.

**23:25** · And basically what you say is that there's some batching dimensions.

**23:30** · In this case, it would be these first two.

**23:33** · And then there's some hidden dimension which doesn't appear on the right side, which means that it gets summed.

**23:40** · And here, I put sum, which means that the aggregation reduction operation is sum.

**23:44** · But you can replace it with mean or max or min.

**23:47** · Yeah.

**23:50** · Is there some speed-up to this?

**23:51** · This basically reduces to the same type of primitive operations.

**23:55** · You can think about it as just sugar.

**23:57** · So it should be the same.

**24:03** · OK.

**24:03** · So the final thing I'll talk about is rearrange.

**24:06** · So this is, I think, a pretty powerful tool.

**24:10** · I think it will come up in an assignment once.

**24:13** · So sometimes you have a dimension that actually represents two dimensions, and you want to operate on one of them.

**24:21** · And the reason this happens is that sometimes you have a matrix and you flatten it.

**24:25** · And then you want to maybe unflatten and flatten.

**24:28** · So this is the way that it works here.

**24:32** · So imagine I have a matrix 3 by 8 but where this dimension, 8 actually represents a 2 by 4 matrix.

**24:46** · So I want to multiply that 2 by 4 matrix by this 4 by 4 matrix.

**24:53** · So what I'm going to do, actually, is to call rearrange.

**24:59** · And what I am doing here is saying, look, this is some number of batch dimensions, which here just corresponds to the first element here.

**25:10** · And then here, I use parentheses to say that this h actually represents the product of heads and hidden1, where I have to-- obviously there's multiple ways to decompose this.

**25:22** · It could be 2 by 4, 4 by 2.

**25:23** · So I said the number of heads is 2, which means that hidden1 is 4.

**25:27** · And then I can break that up into two dimensions, heads and hidden.

**25:35** · OK.

**25:35** · So this creates this.

**25:38** · So before x looked like this, and now x looks like this.

**25:45** · So I guess this might be a little bit hard to see.

**25:50** · So then you can perform your operation transformation on w.

**25:55** · And this is what we've seen before where you have just a standard MatMul where there's some number of batching dimensions for x here.

**26:07** · And this is hidden times hidden by hidden2.

**26:12** · And then once you've done that transformation, you can rearrange it back.

**26:17** · And this is straightforward.

**26:18** · You basically look at two dimensions, and then you can group them into one dimension.

**26:30** · Yeah.

**26:31** · Could you take a two-dimensional thing and shape it into one dimension, aren't there two ways?

**26:35** · Like, you can do it row major or column major?

**26:38** · Yeah, so the question is if you have a one-dimensional thing and you shift it into-- sorry, you have a two-dimensional thing and you shift it into one-dimensional thing, which way do you do it?

**26:47** · Well, the order you do it is specified in the order here.

**26:51** · OK.

**26:52** · Yeah.

**26:57** · So sometimes I find it takes a bit of time to get it used to, but it's well worth it because once you have an einsum, you just think in a different way.

**27:07** · And all the transposes and reductions, all that, it's just it becomes more fluid.

**27:13** · You have to think through these more bespoke primitives.

**27:19** · All right.

**27:20** · So OK.

**27:21** · We're going to use einops.

**27:24** · We'll see a little bit later.

**27:27** · So now let's go return back to the resource accounting question.

**27:30** · So I have tensors.

**27:32** · We've talked about how they take memory.

**27:34** · So how much compute do they take?

**27:36** · So the thing we're going to use to measure computation cost is the number of FLOPs.

**27:43** · A FLOP is a floating-point operation.

**27:45** · And we're going to assume it's a basic operation like addition or multiplication.

**27:50** · So now there's other things that GPUs can do.

**27:53** · But for the most part, we're just going to ignore them because these are the bread and butter, and are going to eat up most of your time.

**28:03** · So one thing that is a pet peeve is that if I say the word FLOPs, it's actually ambiguous what I mean.

**28:11** · So there's FLOPs, which is saying the number of floating-point operations, usually written FLOPs with a lowercase s.

**28:20** · This is a measured amount of computation done.

**28:22** · And then there's FLOP/s, which is floating-point operations per second.

**28:28** · Sometimes it's also very confusing and written as FLOPS with uppercase S, which-- but I'm going to always write /s to make it clear that this is measuring the speed of hardware.

**28:38** · So if you go and see that H100s have 800, 989 teraflops, it's the second.

**28:46** · And when I say that GPT-3 took 325 or whatever it is FLOPs, that's the former.

**28:54** · So just to get that out of the way.

**28:58** · So just to give you an order of magnitude.

**29:00** · So the number of FLOPs when I talk about 1e22 or 23 or 25, these are referring implicitly to the amount of compute or the scale of some of these models.

**29:13** · And so if you look at H100s-- if you look at this glossy spec sheet-- actually it's not on this page.

**29:25** · OK.

**29:25** · Forget it.

**29:27** · There is a spec sheet and I'll tell you that for bf16, the number of FLOP/s is 1979.

**29:40** · And then you go and you benchmark it, and it's like wait a minute, that's not actually what I'm getting.

**29:45** · And then you go read the fine print.

**29:47** · And there's a footnote that says this is with sparsity.

**29:50** · So sparse matrix and for dense over 2.

**29:54** · So you always have to take these numbers divide by 2.

**29:57** · So that's why you see these divide by 2.

**30:01** · OK.

**30:04** · So this allows us to-- just for intuition, so if you have 8 H100s so there's one node for two weeks.

**30:15** · That's 8 times the number of seconds per in two weeks.

**30:23** · Actually, this looks like it's one week.

**30:26** · OK fine.

**30:26** · It's one week times the number of FLOPs you get per second.

**30:30** · So that's about 5e 21.

**30:35** · OK.

**30:36** · So this is just of building intuition for the number of FLOPs that certain types of hardware have and how many FLOPs certain types of models require.

**30:44** · It's nothing fancy.

**30:46** · It's just math and napkin math.

**30:49** · OK.

**30:50** · So now let's do something more mechanical.

**30:54** · So suppose you have a linear model.

**30:59** · It turns out that a lot of this calculus of counting FLOPs is actually going to be at the core it's like linear MatMuls.

**31:08** · So this is actually not without too much loss of generality.

**31:11** · So you have n points.

**31:12** · Each point is d-dimensional.

**31:14** · And we're going to map each of these d-dimensional vectors to a k-dimensional output.

**31:25** · So B is going to be the number of points.

**31:28** · D is the number of dimension, input dimensions, and K is the number of output dimensions.

**31:32** · OK.

**31:33** · So let's construct some x, which is the data matrix B by D. The weight matrix is D by K.

**31:42** · And when you do the MatMul, the question is how many FLOPs that is.

**31:49** · And it turns out that this is going to be 2 times basically the product of all the three dimensions.

**32:00** · And the way to see that is that we have one multiplication for each triple and then also one addition.

**32:11** · So there's a minus 1, because if you don't have to actually add like in this case D minus 1 times.

**32:20** · But let's ignore that.

**32:22** · OK.

**32:23** · I'll come back to this.

**32:24** · If this was a bit fast, I think there's another way to derive this.

**32:31** · So what about the FLOPs of other operations?

**32:34** · So elementwise operations are just the size of a matrix.

**32:39** · I think that's fairly clear.

**32:40** · So addition also requires m n FLOPs.

**32:44** · So in general, no other operation you'll encounter as expensive as matrix multiplication for large enough matrices.

**32:51** · So in general, we're just going to focus on what MatMuls are doing, with important caveat of when we talk about memory.

**32:58** · Yeah.

**33:00** · Just from interest, there are some other algorithms for doing matrix multiplication.

**33:06** · Is this only for just doing it normally?

**33:10** · So the question is that there are other algorithms of doing matrix multiplication.

**33:13** · You mean like sub cubic algorithms?

**33:16** · Yes.

**33:17** · So in general, that's-- the optimization, the algorithms that people are going to explore for multiple matrix multiplications are going to be much more about how you co-design with the systems, rather than these more asymptotic algorithms.

**33:32** · Yeah.

**33:33** · Yeah.

**33:34** · We're considering additional multiplication in the same way is not possible to do division more efficiently than multiplication.

**33:42** · Yeah, so I think the way the hardware is built, the two are basically the same.

**33:48** · But yeah, intuitively it seems like I can do addition faster than I do multiplication.

**33:53** · But the way the hardware is kind of the same.

**33:59** · OK.

**33:59** · So you can think about this MatMul as B is the number of data points.

**34:07** · And D K is the number of parameters.

**34:09** · So remember x is B by D and w is D by K.

**34:14** · So another way to think about this formula is that the number of FLOPs in order to do a forward pass of this linear matrix is actually 2 times the number of tokens or data points times the number of parameters.

**34:33** · So it turns out that this actually generalizes to transformers, which if you remember the 6 times n times d formula, we can see the shape of that forming.

**34:45** · OK.

**34:46** · So unfortunately these calculations are not going to be very meaningful because I'm doing this on CPU.

**34:54** · But I'll just walk through the code here.

**34:58** · So what we've so far done is measured FLOPs.

**35:04** · So this is independent of hardware.

**35:07** · It's just like the number of calculations you need to do for your model.

**35:12** · So now the question is, how long does it actually take on hardware?

**35:15** · So one way to find out is you just time it.

**35:18** · So in this class, I think in the few lectures, we're going to talk more about benchmarking.

**35:24** · But here's a little preview.

**35:27** · So in general, when you time especially on GPU, you have to call cuda synchronize to make sure that-- because the GPU is running asynchronously, you want to make sure that you have this synchronization point.

**35:46** · And then you perform the operation.

**35:47** · And after operation, then you also have to have the synchronization barrier.

**35:55** · If you omit this, you're going to find that wow, your timings are really fast.

**35:59** · And that's because this is a non-blocking call.

**36:01** · It just returns.

**36:02** · And often it's general good practice to try this multiple times and take the average.

**36:11** · So the actual FLOPs per second is basically the number of FLOPs you did times the time that you recorded on your hardware.

**36:24** · And so remember, there's also another number, which is the GPU has a spec sheet.

**36:32** · Let's see if I can pull it up on this link.

**36:39** · I feel like maybe I have the wrong link.

**36:41** · I'll have to fix that after class.

**36:42** · --that it gives you some number of FLOPs, which was 989 teraflops.

**36:54** · And so in general, the number of actual FLOPs per second is going to be different from the promised FLOPs per second.

**37:07** · OK.

**37:08** · Oops, sorry.

**37:12** · And the way to directly think about the discrepancy is something called Model FLOPs Utilization, or MFU.

**37:20** · And the definition of MFU is the actual FLOPs per second divided by the promise of FLOPs per second.

**37:27** · And here this is ignoring the communication and other overhead.

**37:34** · So you basically take the actual and divide by the promise.

**37:39** · So in general, it's rare that you get more than was-- I mean, it's-- yeah, you just never get anything more than what you were promised.

**37:50** · Often you get less.

**37:52** · And in general, if you get about MFU of 0.5 for modern models, you should be pretty happy with yourself.

**38:01** · If you have just like a straight up MatMul, you can get maybe potentially like 80.8 even, but you usually can't get that high.

**38:12** · And sometimes if you have really-- something's really wrong, you'll get something like 0.1, which means that you should do something.

**38:21** · OK.

**38:22** · So whenever you write your model, you can calculate.

**38:27** · You now know how to calculate the MFU through a combination of counting the number of logical FLOPs that your model needs to do, and then looking at the wall clock time and essentially dividing.

**38:41** · Yeah.

**38:42** · Was there a question over there?

**38:43** · Oh.

**38:44** · \[INAUDIBLE\] Yeah.

**38:49** · So is the promise-- what is the promise FLOPs number?

**38:53** · That is in the spec sheet, it is already divided by 2 of the 989 number.

**39:02** · And then on top of that, you only get 0.5 of that in general depending on your computation.

**39:10** · \[INAUDIBLE\] you're getting 50% MFU?

**39:15** · Yeah, so why are you getting only 50% MFU?

**39:18** · Actually, that's a good question.

**39:19** · I'll come back to that when we talk about memory bottlenecks.

**39:26** · OK.

**39:27** · So to summarize here, matrix multiplications dominate generally the computations.

**39:35** · And that's by design.

**39:41** · And the number of FLOPs per second depends on the hardware.

**39:46** · So better hardware leads to more FLOPs and also it depends on the data type.

**39:54** · Which means that if you look at the spec sheet, different data types will have different FLOPs.

**40:00** · If you try to do float32 nowadays, it's going to be really, really slow because they're not really optimizing for that workload.

**40:07** · Whereas now bf16 or fp8 are going to be much faster.

**40:16** · And MFU-- now you know what MFU is.

**40:18** · It's the actual FLOPs divided by promised FLOPs.

**40:27** · All right.

**40:28** · So to go back to the question of why is MFU 0.5.

**40:32** · And to understand that, I'm going to have to introduce this idea of arithmetic intensity.

**40:37** · And the reason is that, well, it's not just doing a bunch of MatMuls and then you're done and looking at how long the MatMuls take.

**40:50** · This is my very cartoon version of what hardware looks like.

**40:56** · You have high bandwidth memory.

**40:58** · And then you have where the compute cores are, the accelerators chips are.

**41:04** · And then how do you compute?

**41:08** · Well, you have to send your inputs, your matrices-- the tensors are sitting down here-- from the memory to the accelerator.

**41:17** · You do the computation, and then you send it back.

**41:22** · So if you want to measure how long this takes, this depends on two things.

**41:27** · One is the accelerator speed, which is what we've talked about just now.

**41:34** · But the other thing that matters is the memory bandwidth of your hardware, which we haven't talked about.

**41:39** · And if you look at the spec sheet, we talked about how the FLOPs per second was 1979e12 divided by 2.

**41:50** · And the bytes per second, which is the memory bandwidth, this is 3.3 terabytes per second.

**41:57** · And remember why we were looking at memory, how much things took to store.

**42:07** · It's most obviously that, well, if you have a model that's too big it doesn't fit in your memory, that's not going to be fun.

**42:15** · But also it turns out that memory-- you need to move this memory, which takes time.

**42:21** · So actually the size of how large things actually influences speed as well.

**42:27** · OK.

**42:27** · So I'm going to talk through some operations, and I'm going to basically compute how long things are going to take and introduce this idea of arithmetic intensity.

**42:38** · OK, so suppose I have a million dimensional vector of bf16.

**42:45** · And I'm going to just compute a ReLU on this.

**42:49** · So remember ReLU is just max of x and 0 done elementwise on the entire vector here.

**42:58** · So I count two things, one is the number of bytes that were moved.

**43:05** · So I have to read x in-- copy it into the accelerators.

**43:14** · And each of this is going to be 2 because bf16 is 2 bytes per float times n floats.

**43:21** · So that's 2n.

**43:22** · And then I'm going to write y back.

**43:25** · So that's another 2n.

**43:27** · OK.

**43:27** · So that's the number of bytes that have to be moved.

**43:30** · And then how many FLOPs were done?

**43:32** · Well, each of these elements, I'm just comparing it with zero, and that's it.

**43:38** · So that's n comparisons.

**43:41** · So now I look at the communication time, which is the number of bytes that I needed to move divided by the speed of that movement.

**43:55** · And that gives me the time, which is 1e minus 6 seconds.

**44:01** · And what about the computation time?

**44:02** · That's the FLOPs divided by FLOPs per second.

**44:05** · So that's 1 minus 9.

**44:09** · OK.

**44:10** · So there's also going to be another important assumption which generally, we try to hold is that we overlap communication and computation.

**44:19** · We're going to talk more about that when we talk more deeply about GPUs.

**44:23** · But the idea is that in this case as we don't sit here waiting for the things to move.

**44:32** · As soon as they're there, we start computing them, and then we move them back.

**44:36** · So this movement and also the compute is happening at the same time.

**44:43** · So mathematically, we're just going to assume that the total time is the max of the two, because we're going to assume that we can perfectly overlap them.

**44:52** · In practice, it's not going to be perfectly overlapped.

**44:54** · There's going to be some overhead.

**44:55** · But this is good enough for now.

**44:58** · OK.

**44:58** · So the total time, as we see here, is 1e minus 6.

**45:04** · OK.

**45:05** · So if you ask what is the bottleneck here?

**45:08** · So when the communication time is greater than the computation time, then we call it the algorithm memory bound, because you're spending most of your time just waiting for bits to show up.

**45:19** · And when the computation time is greater and the communication time, that's compute bound, because then you're actually-- your bottleneck is actually doing the compute.

**45:29** · And so in this case, what is ReLU?

**45:34** · Rectified Linear Unit.

**45:35** · Oh, sorry.

**45:36** · Is it memory bound or compute bound?

**45:39** · Memory bound.

**45:40** · Memory bound.

**45:41** · Yeah.

**45:44** · And it's clear, because the compute is way less than the communication.

**45:53** · So here's another way to see it.

**45:56** · And this is where I'm going to define the intensity.

**46:00** · So what is the intensity of an accelerator is essentially how much work can the accelerator do per byte transferred.

**46:08** · And for any given accelerator based on the spec sheet, you basically have the FLOPs per second divided by the bytes per second.

**46:20** · So how much useful work can you do per byte that's moved for each 100?

**46:26** · That's 295.

**46:28** · So that means for every byte of-- you can do 295 floating point operations.

**46:38** · So that's an intuitive number to have in your head, about 300.

**46:43** · OK.

**46:43** · So now the arithmetic intensity of algorithm is how much actual work was done per byte for this workload.

**46:51** · And if you look at it for the ReLU computation, it's FLOPs over bytes and it's actually-- so this is actually a quarter, I guess, not half.

**47:02** · So the point is that it's very small, 0.25.

**47:07** · OK.

**47:08** · So now we can talk about bottlenecks through the language of intensity.

**47:12** · So something is memory bound if the arithmetic intensity is smaller than the accelerator intensity and compute bound if it's greater than accelerator intensity.

**47:25** · So these are equivalent.

**47:26** · And if you look at the algebra, it's basically you have two fractions.

**47:29** · And then you just multiply and divide to-- you basically switch the terms around.

**47:35** · OK.

**47:36** · So in this case, we're memory bound.

**47:42** · So in general, we're going to find ourselves in a situation where we're memory bound because data movement is expensive.

**47:49** · And so if you can get higher arithmetic intensity, that's good.

**47:54** · So 0.25, if you see that number, if someone tells you arithmetic density is 0.25, you should say, oh, this is really bad.

**48:03** · Yeah.

**48:03** · What's some typical arithmetic intensity for some transformer \[INAUDIBLE\]?

**48:08** · Yeah, so we'll get to that.

**48:09** · OK.

**48:11** · All right.

**48:12** · So OK.

**48:14** · So one way to think about increasing arithmetic intensity is that let's just try to do more stuff per unit of byte moved.

**48:23** · So the GELU is another activation.

**48:26** · It looks like this, some formula that is more-- doesn't have zeros.

**48:35** · And if you do this calculation, the number of bytes are moved back and forth is still 2n plus 2n.

**48:45** · And the FLOPs here it's about 20 FLOPs per elementwise scalar operation, so it's 20n.

**48:54** · So the arithmetic intensity is, let's say, 5.

**48:57** · This is a crude estimate.

**48:59** · And in this case, are we memory bound or compute bound?

**49:09** · Memory bound.

**49:10** · Memory bound, because 5 is still smaller than 295, way smaller.

**49:17** · So even though GELU does a lot of work, more work than ReLU, in the way that things are structured, it's still memory bound.

**49:30** · Which means that if you were just computing ReLU and GELU, you think that, well, GELU is so complicated, it must be really expensive.

**49:38** · But actually it's exactly the same, because that's not where the bottleneck is.

**49:46** · OK, so now let's look at some linear operations, so dot product.

**49:50** · So dot product, you have a vector x.

**49:53** · You have a vector w, size n, and you take the dot product.

**49:56** · So how many bytes are moved?

**49:58** · So you read x, which is 2n; read w, which is 2n; and then you write y, which is a scalar, which is 2.

**50:05** · And the number of FLOPs is you do the n multiplications and n minus 1 additions.

**50:13** · So that's 2n minus 1.

**50:16** · OK.

**50:16** · So what's the arithmetic intensity?

**50:19** · Oh, for this one, it's about half, which is also pretty bad.

**50:24** · Which means that-- hopefully you get the idea-- it's memory bound.

**50:33** · So what about a matrix vector product?

**50:36** · So I have this x is a vector. w is an n by n matrix.

**50:41** · And you form this product.

**50:43** · How many of you think this will be compute bound?

**50:46** · How many of you think memory bound?

**50:48** · OK.

**50:48** · So let's see.

**50:51** · So I'm going to read x read, which is 2n; read w, which is 2n squared; and write y, which is 2n.

**50:59** · And there's the number of FLOPs is basically you're doing n dot products.

**51:05** · So that's n times the cost of doing a dot product.

**51:09** · And the arithmetic intensity is barely, barely higher.

**51:16** · So this is also memory bound.

**51:21** · OK.

**51:21** · So now let's talk about matrix multiplication.

**51:25** · So this is where things get interesting.

**51:27** · So I have an n by m matrix, another m by n matrix.

**51:30** · And I multiply them.

**51:31** · And the number of bytes that were moved was 2n squared plus 2n squared.

**51:39** · And then I have to write 2n squared bytes back to y.

**51:43** · And the number of FLOPs here is n squared dot products.

**51:49** · And then what's arithmetic intensity?

**51:52** · It's 300.

**51:53** · Whew.

**51:53** · OK, so 340.

**51:56** · And in general, it's roughly n over 3.

**52:01** · And intuitively this makes sense because you're sending n squared things, but you're computing n cubed things.

**52:10** · So the number of things you're computing over a number of things you're sending is order n.

**52:16** · And this gets better the larger you make the matrices.

**52:19** · So in general, this is why when you hear people talk about, oh, we need to make large batch sizes or have large matrices, it's exactly this.

**52:29** · If you're under the accelerator intensity, making things smaller doesn't actually speed things up.

**52:38** · It's all the same.

**52:39** · Whereas if you get to a point where you're over this, then you're actually saturating your GPUs.

**52:47** · OK.

**52:48** · So finally, this is compute bound.

**52:52** · OK.

**52:53** · So as long as we have large matrices, we're actually pretty good.

**52:56** · We're in a compute bound saturating the accelerator.

**52:59** · And the question earlier about what about transformers?

**53:03** · Well, it turns out we'll see in both your assignment, but also in the next lecture that transformers are essentially big matrix multiplications with some things sprinkled in between.

**53:14** · So that's good news from arithmetic intensity.

**53:19** · And this is, by design, transformer is designed in a certain way to have high arithmetic intensity.

**53:26** · And one comment here, just to foreshadow the inference lecture, is that matrix vector products is essentially what goes on when you're doing transformer inference.

**53:36** · Because inference, you're generating one token at a time.

**53:39** · And so you only get to-- it's like a vector that you're trying to dot product with a matrix.

**53:45** · And that's as we saw with memory bound.

**53:49** · Whereas at training time, you get this whole sequence and you're processing all at once.

**53:56** · OK.

**53:58** · So note also that the intensity depends on the precision.

**54:03** · So by default everything we're doing here is fp16.

**54:08** · OK.

**54:09** · So also to tie it to the other question about MFU, and the reason you might be getting low MFU is that while you might be doing-- so MFU is a promise FLOPs-- sorry, actual over promise.

**54:28** · So if you have really large memory bottlenecks, then you're not actually going to get very good throughput, even though you-- the promise is if you didn't have memory bottlenecks, you were just like going through and doing all the computations of your model.

**54:47** · OK.

**54:48** · Final thing on arithmetic intensity roofline plots.

**54:53** · So there's a nice way to-- yeah.

**54:54** · \[INAUDIBLE\] generally speaking, these models tend to be memory bound, as they say for most of the computation.

**55:04** · But yet the accelerators are 50% is not as good.

**55:08** · And maybe you get 70, maybe you get 80 that's really good, meaning that the accelerator is outsized compared to memory bandwidth.

**55:18** · Why is it like that?

**55:20** · Why do these accelerators-- are just big while they're just idling or waiting for memory?

**55:27** · So the question is, could you design maybe accelerators that had better characteristics?

**55:35** · Yeah.

**55:36** · Maybe when we talk about GPUs and understand a bit more how they work, we can talk about why this is.

**55:42** · But if you have an answer, you should tell Jensen.

**55:44** · And maybe you can design a better hardware.

**55:49** · OK, so let's visualize the relationship between arithmetic intensity and performance.

**55:55** · So this plot basically plots the arithmetic intensity on the x-axis.

**56:00** · So every slice here corresponds to a particular, let's say, algorithm.

**56:06** · And then we have these lines here.

**56:09** · And each of these lines corresponds to, let's say, a particular accelerator.

**56:13** · Maybe H100 or B200 or so.

**56:18** · And so what this shows is-- and then on the y-axis is the FLOPs per second that are realized.

**56:26** · So if your algorithm has low arithmetic intensity, like, ReLU or dot products, you're going to be over here, which means that the FLOPs per second realize is going to be not as high as your peak accelerator.

**56:42** · And as you increase the memory arithmetic intensity, that's going to-- things are going to be better.

**56:49** · You're going to be able to saturate your hardware up until a certain point.

**56:55** · And after a certain point your compute bound.

**56:57** · And obviously you can't exceed the peak FLOPs.

**57:04** · All right.

**57:05** · So let's go on here.

**57:09** · OK.

**57:10** · So now I'm going to go back and talk about memory and compute for the operations that we need in training.

**57:22** · So far we've done tensor operations, basically MatMuls.

**57:26** · And we saw basically how much memory it took and how much compute it took and the interaction between them.

**57:34** · So now let's actually think about what it takes to train.

**57:37** · So here is our running example here.

**57:41** · Actually it's not a linear network, just a deep network.

**57:45** · So I'm going to consider a case where I have an input, which is a B by D input, and a number of layers where each layer is a D by D MatMul.

**57:59** · And then that produces some set of preactivations and then element wise ReLU that produces some activations of the first layer.

**58:08** · And then this is repeated again and again.

**58:12** · And the output is just going to be the same size.

**58:17** · So the deep network, I mean, just to see what it looks like in PyTorch, just so it's basically a set of blocks.

**58:25** · Each block has a weight vector-- sorry, a weight matrix associated with it.

**58:34** · And so when you the number of parameters here is D squared times the number of layers L.

**58:45** · And when you run the model on the batch of data, what we're doing is we're going through the layers, and each layer we basically apply the linear transformation and then a pointwise ReLU activation.

**59:02** · And then we do that for all the layers.

**59:04** · So this is hopefully straightforward very simple model.

**59:09** · OK.

**59:10** · So now let's talk about gradients.

**59:15** · So let's use actually even simpler example.

**59:19** · So this is just a simple linear model regression.

**59:24** · So we have a vector 1, 2, 3, a weight vector 1, 1, 1.

**59:33** · And we take the dot product, and we form this MSE loss.

**59:38** · OK.

**59:38** · And what happens when we take the-- do the backward pass is that each of the variables involved in this computation graph has a gradient that is either set or not set.

**59:53** · So the w.grad gets set to 1, 2, 3.

**1:00:01** · So this is just basic mechanics of PyTorch.

**1:00:04** · I think it should be familiar to all of you.

**1:00:08** · So now the question is, how much compute does gradients take?

**1:00:15** · OK.

**1:00:15** · So let's count FLOPs for computing gradients.

**1:00:21** · So let's take a simplified model where you have an x, which is BID and times w1 matrix.

**1:00:33** · We're going to ignore this ReLU for now just for simplicity times w2, which is the same D by D-- same shape D by D matrix.

**1:00:41** · OK.

**1:00:42** · I'm going to use einsum here for reasons that will become clear.

**1:00:47** · So the first thing you do in this linear network is you take x and you take w1. x is batch by input dimension.

**1:00:55** · w1 is as in buy out, and you get a batch buy out.

**1:01:00** · And h2 takes h1 and w2, and it's the same story.

**1:01:05** · These are just MatMuls.

**1:01:09** · And then you form a loss.

**1:01:10** · I'm just sending it to some arbitrary just to get a number out.

**1:01:13** · OK.

**1:01:14** · So what happens in the backward pass, I'm going to call these retain gradients for debugging purposes, which you'll see later.

**1:01:23** · You take the backward pass on the loss.

**1:01:25** · And the question is, how much work did that-- how many floating point operations was that?

**1:01:31** · So let's zoom in on one layer.

**1:01:33** · Let's focus on the second layer here, so this layer.

**1:01:39** · And so the second layer takes h1 and just multiplies it by a matrix w2 to get h2.

**1:01:45** · And that's it.

**1:01:46** · So if you look at the FLOPs in the forward pass, this is just a MatMul.

**1:01:53** · And we just take the three dimensions, and we multiply them together and bf16.

**1:01:58** · That's 2 bytes.

**1:02:00** · Sorry, that's irrelevant.

**1:02:02** · This is just 2 because it's an addition and a multiplication.

**1:02:08** · So in the backward pass, what does this look like?

**1:02:12** · So in the backward pass, if you remember your chain rule and your backprop algorithm, you have to compute two things.

**1:02:18** · You have to compute the backward message, the gradient of the loss with respect to your input.

**1:02:30** · And you also have to compute the gradient with respect to your parameters.

**1:02:36** · And what do those gradients look like?

**1:02:39** · This is just by the definition of these-- I guess this is just the chain rule here written in einsum notation.

**1:02:47** · You take the d loss d h2, which is h2 grad.

**1:02:54** · That's a batch by Alt matrix, and you multiply it by w2, which is in by out.

**1:03:04** · And then that gives you h1 grad, which is d loss d h1 and which is batch n.

**1:03:16** · So I always get this confused whenever if you see-- learn calculus, there's like, one of them has a transpose, and I forget which order to put them in.

**1:03:25** · And einsum I think makes it very clear, because you can remember even just looking at the scalar case that it has to look like h1 grad is h2 grad times w2.

**1:03:37** · And the only question is, how do you index things?

**1:03:40** · And you index things by basically looking at the shape of these-- and the named dimensions.

**1:03:46** · And in this case, it's just a matrix multiplication where you are summing over the output dimension.

**1:03:54** · And when you do that, we can check that.

**1:03:57** · So h1.grad was the thing that was computed by loss.backward.

**1:04:01** · And this is actually indeed the same thing that I wrote out here just as a sanity check.

**1:04:07** · So the other thing you have to compute is w, the gradient with respect to the parameters.

**1:04:13** · And that's d loss d h2 times h1.

**1:04:18** · So it's the same kind of backward message that's coming back but multiplying with the other thing that you're not taking the gradient with respect to.

**1:04:27** · And you just write out the dimensions.

**1:04:30** · And in this case, you're summing over the batch dimension.

**1:04:40** · But the nice thing is that if you just look at this, we know how expensive this is.

**1:04:47** · The number of FLOPs is essentially the product of all the dimensions.

**1:04:52** · It doesn't matter which ones you're batching or not.

**1:04:55** · It's basically you enumerate over the i, j, k, and you aggregate.

**1:04:59** · The way that you aggregate is different, but the FLOPs is the same.

**1:05:05** · OK.

**1:05:05** · So notice that the backward pass is exactly twice as expensive as the forward pass.

**1:05:12** · And this is because you have to take-- compute two gradients, one with respect to the parameters for each parameter, and then one with respect to the input, the other thing that's not the parameter.

**1:05:27** · Maybe I'll pause.

**1:05:29** · Any questions about that.

**1:05:39** · OK.

**1:05:40** · So now that was just for w2.

**1:05:43** · And you just need to apply this to all the parameters in the network.

**1:05:47** · And you put it all together, you will see that for this network, the forward pass is 2 times the number of data points, which is b times the number of parameters FLOPs.

**1:05:59** · And the backward pass is twice of that, which is 4 times the number of data points times the number of parameters FLOPs.

**1:06:05** · And so the grand total is 2 plus 4 is 6 and 6 times the number of data points and parameters.

**1:06:12** · So this is where the 6 nd comes from that you might have seen in various places.

**1:06:18** · It's just by counting forward and backward.

**1:06:24** · So we did this for just these deep networks.

**1:06:28** · But it turns out that this is actually a good approximation for transformers as well-- as long as the context length isn't too large.

**1:06:36** · If the context length is too large, then you get the context length squared, and that's more FLOPs that isn't in this kind of accounting.

**1:06:47** · OK.

**1:06:47** · So let's talk a bit about-- so now we have gradients.

**1:06:52** · Now we have to-- the other piece when you do training is we have to do optimization.

**1:06:57** · So here's our deep network.

**1:07:01** · Just so that I'm not just giving you what's in assignment 1, I'm going to use the AdaGrad optimizer, which this from-- this is from 2011.

**1:07:12** · This is predates Adam.

**1:07:13** · It's where you can think about it as somewhere in between SGD and Adam.

**1:07:18** · And basically it's SGD where you look at the second moments of the gradient.

**1:07:28** · Momentum is what you look at the first moment gradients.

**1:07:30** · And Adam is where combine the two of them.

**1:07:36** · I'm not going to have time to go into the optimizer details here, since we're focusing mostly on the usage.

**1:07:42** · So we can define an optimizer.

**1:07:46** · And when you compute the gradients and you compute the gradients and then when you take an optimizer step, if you're defining a new optimizer, so in the assignment 1, you're going to implement Adam.

**1:08:06** · For each of the parameter groups which are, in this case w1 or w2, we look at-- there's something called optimizer state, which is of storage that you can-- the optimizer uses while it's running.

**1:08:27** · And what we're computing in AdaGrad is something-- is the squared gradients, the sum of the squared gradients.

**1:08:36** · So here we're getting it from the optimizer state.

**1:08:40** · We're updating it with the current gradient, and we're storing it back.

**1:08:44** · And then after you update the g2, then you update the parameters.

**1:08:50** · So in AdaGrad, it's you basically divide by the square root of the average gradient squared or sum of gradient squared, rather.

**1:09:00** · OK.

**1:09:01** · So let's see.

**1:09:06** · So I'm actually going to maybe skip the training loop.

**1:09:13** · Actually, wait.

**1:09:15** · Hold on.

**1:09:16** · I think I actually skipped something I didn't mean to.

**1:09:19** · OK.

**1:09:20** · So that was the optimizer state, and now let's look at how much memory the optimizer is using.

**1:09:27** · Or in general what is the memory usage?

**1:09:29** · So for parameters in this network, there's D squared parameters per each of the L layers.

**1:09:37** · And each parameter takes 2 bytes if we're storing it in fp16.

**1:09:41** · So that's the number of parameters.

**1:09:42** · Activations-- this is 2 times-- which is for bf16 batch times D times the number of layers.

**1:09:55** · For every layer you have activation.

**1:09:57** · You have gradients which is basically a copy of all the parameters.

**1:10:02** · And this is also bf16.

**1:10:04** · And the optimizer state is for every-- actually this should be the-- I think this should be-- sorry, I'll fix this.

**1:10:12** · This is a typo.

**1:10:13** · This should be the number of parameters, not parameter memory.

**1:10:16** · And this should be number of parameters, so number of parameters times 2, and then this is 4 times the number of parameters.

**1:10:24** · And the reason for this is that it's customary to use fp32 for the optimizer states for stability reasons.

**1:10:31** · Obviously people have tried using bf16.

**1:10:35** · And what ends up happening is that you're taking squares, and you're averaging multiple steps, and it doesn't-- it's not very stable.

**1:10:45** · And for AdaGrad, it's here-- we have 4 bytes per parameter for storing the optimizer state.

**1:10:50** · Adam-- you store the first order bit and the second order moments, so that's 8 bytes.

**1:10:56** · So if you think about the optimizer states is actually a lot of the memory used.

**1:11:04** · One note is that remember memory serves two purposes.

**1:11:09** · One is how much-- you have to store this thing in your HBM.

**1:11:12** · But the other thing is that it has to be shipped to the accelerators.

**1:11:22** · And in general, the optimizer state is not really the bottleneck for compute.

**1:11:28** · So the amount of memory here is not really so important for performance in terms of speed, but it means that you can't fit large models in your memory.

**1:11:42** · OK.

**1:11:42** · So to put it together, as we mentioned, the number of parameters here is D times D times L, and the number of FLOPs is D times number of tokens or number of data points times number of parameters.

**1:11:55** · So for transformers, this is going to be a bit more complicated.

**1:11:59** · But you're going to do that in assignment 1.

**1:12:01** · And you do it more carefully.

**1:12:03** · OK.

**1:12:04** · I'm going to skip the training loop.

**1:12:05** · This is just a general review.

**1:12:10** · Two things I want to quickly touch on before we conclude.

**1:12:12** · One is that as we see, memory does have an important effect both on the ability to store large models, but also sometimes on your speed.

**1:12:24** · And so in general, you want to reduce your memory usage.

**1:12:26** · So there's two things that people typically do.

**1:12:30** · One is gradient accumulation.

**1:12:34** · So in general, you want to use batch sizes that are large enough to improve stability up to a critical batch size, which Tatsu will talk about later.

**1:12:45** · And as we saw that activation memory scales with batch size.

**1:12:50** · So you might want to run out of memory at some point if you have two large batch sizes.

**1:12:54** · So gradient accumulation says, well, you compute.

**1:12:58** · You have these micro batches, and you compute the gradient on the micro batches.

**1:13:02** · And then you just accumulate the gradients.

**1:13:04** · You don't zero out the gradients.

**1:13:05** · And every batch size over microbatch size steps, you update the parameters and zero out the gradients.

**1:13:12** · OK, so this is actually a very simple code change which allows you to save on compute.

**1:13:19** · So the other thing I'll quickly mention is activation checkpointing.

**1:13:23** · So in training, we need to store in general the activations of all the layers.

**1:13:31** · This is done by default. Actually, interestingly, for inference we don't need to compute the gradient, so we only need to store the current layer's activations.

**1:13:40** · But for training, the memory usage is B times D times 2 times L.

**1:13:52** · And the question is, can you reduce this?

**1:13:56** · So activation checkpointing, also known as gradient checkpointing or rematerialization-- the key idea here is that in the forward pass, you just keep activations for only a subset of the layers.

**1:14:07** · And in the backward pass, you recompute the missing activations from the last checkpoint.

**1:14:12** · That's why it's called checkpointing.

**1:14:14** · And this is a general trick that you see in systems which is trading off.

**1:14:19** · If you want to reduce memory, you can just recompute things.

**1:14:23** · So here we're going to define this to operationalize this.

**1:14:27** · This is actually fairly easy.

**1:14:33** · Actually, I didn't-- OK, maybe I didn't see the-- OK, so basically what you do here is you have the same model, except for you just add Torch utils at checkpoint on a layer.

**1:14:47** · So that means do this computation, but don't store any of the intermediate activations.

**1:14:54** · Only store what is needed.

**1:14:58** · So this means that if you store all the activations, remember in your deep network, you have the thing that's pre the ReLU and the thing that's after the ReLU right.

**1:15:10** · So that's a lot of storage.

**1:15:12** · And instead if you do activation checkpointing on those blocks where each block has a linear and ReLU, you don't store the pre ReLU, and you save basically half.

**1:15:22** · You can get away with half the memory.

**1:15:24** · And then when you're doing the backward pass, you need g3, but you can compute g3 easily from h2.

**1:15:32** · OK.

**1:15:33** · You can go further than this.

**1:15:36** · You can say, well, I can store all the layers, but in the extreme case, I can just store no layers.

**1:15:45** · And so that will be maximally memory efficient.

**1:15:49** · The only thing is that the compute is going to be L squared.

**1:15:52** · Because for every one of these layers, you have to start from the beginning.

**1:15:57** · And maybe a sweet spot is if you store the checkpoints at square root L layers, that means your activation memory is square root of L, and your recomputation overhead is also square root of L. So that's balanced.

**1:16:15** · OK.

**1:16:16** · So to summarize this lecture, everything is operating on tensors-- parameters, gradients, activations, optimization states, data.

**1:16:25** · We introduced this einops, which hopefully you guys can embrace as a way to think about tensor operations.

**1:16:30** · 6 times number of data points times number of parameters is a formula which now we have demystified as the number of FLOPs per-- for training step.

**1:16:41** · And actually if this is training step, this should be batch size.

**1:16:46** · And then we talked about arithmetic intensity and roofline analysis, which allows us to diagnose whether a computation is memory bound or compute bound.

**1:16:56** · Matrix multiplications are compute bound.

**1:16:59** · Basically everything else is memory bound.

**1:17:01** · And then finally, gradient accumulation, activation, checkpointing are ways to reduce the memory.

**1:17:07** · And by reducing the memory, that allows you to use bigger batch sizes.

**1:17:13** · OK.

**1:17:14** · So that's it for today's lecture.

**1:17:16** · So next week, Tatsu will talk about architectures.
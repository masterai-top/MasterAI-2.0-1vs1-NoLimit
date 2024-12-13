德州1对1的AI，德州AI人工智能；MasterAI-2.0-1vs1-NoLimit

## Introduction

MasterAI v2.0 is an iterative algorithm derived from MasterAI v1.0 
It utilizes profound Reinforcement Learning + Search in imperfect-information games and achieves superhuman performance in heads-up no-limit Texas Hold’em. Furthermore, it is a major step toward developing technologies for multiagent interactions in real world.
MasterAI v2.0是从MasterAI v1.0衍生出来的迭代算法，它在非完全信息游戏中利用了通用的强化学习+搜索，并在一对一无限押注的德州扑克中实现了超人的表现。此外，这是在现实世界中开发多智能体交互技术的重要一步。可以应用在线下和线上的poke游戏等各种场景；

## Technology

1.MaterAI v2.0 algorithm generalizes the paradigm of self-play reinforcement learning and deep learning and search through gargantuan imperfect-information. It makes decisions by factoring in the probability distribution of different beliefs each player might have about the current state of the game and uses counterfactual Regret minimization (CFR) algorithm to search efficiently.


2.Our experiments confirmed that MasterAI does indeed converge to an approximate Nash equilibrium in two-player zero-zum game

## Technical bottlenecks

Some technical bottlenecks are encountered when training the algorithm model with CFR framework. For instance, the large state space is leading to too much computation:

1.Algorithm training has a large amount of calculation (2560000 * 1750 in the paper)

2.Deployment speculation and search time is too much: 3 ~ 5 seconds

3.The number of nodes in Abstract CFR (400BB) Betting Tree is too large, more than 400 million

## Contact us

The Master team is constantly exploring the innovation of AI algorithm, and hoping that like-minded technical experts from all over the world can communicate and exchange here, or join us to make MasterAI bigger and stronger together. Please feel free to contact us at Telegram：@alibaba401

                                                                MasterAI v2.0
一、简介

 MasterAI v2.0是从MasterAI v1.0衍生出来的迭代算法，它在非完全信息游戏中利用了通用的强化学习+搜索，并在一对一无限押注的德州扑克中实现了超人的表现。此外，这是在现实世界中开发多智能体交互技术的重要一步。
 ![微信图片_20241030103018](https://github.com/user-attachments/assets/a68c45e7-a4f5-4241-a85d-0a9cb7a85546)

 二、运用技术
 1） 一种将自我博弈强化学习和搜索相结合推广到不完美信息游戏的算法；
 2） MasterAIv2.0是源自MasterAI v1.0的迭代算法，在不完全信息游戏中实现通用的强化学习+搜索算法，该算法为一种通用的人工智能框架，基于公共置信状态 ，通过单层前瞻搜索，MasterAI通过考虑每个玩家对游戏当前状态可能拥有的不同置信的概率分布来做出决策,评估一对N的可能性;
 3）MasterAIv2.0运用反事实遗憾最小化（CFR）,这是一种在双人零和博弈中收敛至纳什均衡的迭代算法,利用折扣原则（discounting）来显著加快收敛速度；
 4） 深度神经网络。
 
 三、瓶颈使用CFR框架训练算法模型时会遇到一些技术瓶颈 。
 例如，大的状态空间会导致过多的计算：
 1）算法训练计算量大（论文中2560000 * 1750）；
 2）部署推测和搜索时间过多：3 ~ 5 秒瓶颈 ；
 3） Abstract CFR (400BB) Betting Tree的节点数量过大，超过4亿 。


Master AI 2.0 的训练模型数据以及核心算法代码有偿出售。有兴趣者联系：telegram：@xuzongbin001

 

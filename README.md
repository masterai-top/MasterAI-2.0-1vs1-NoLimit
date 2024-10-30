MasterAI-2.0-1vs1-NoLimit

## Introduction

MasterAI v2.0 is an iterative algorithm derived from MasterAI v1.0 
It utilizes profound Reinforcement Learning + Search in imperfect-information games and achieves superhuman performance in heads-up no-limit Texas Hold’em. Furthermore, it is a major step toward developing technologies for multiagent interactions in real world.
MasterAI v2.0是从MasterAI v1.0衍生出来的迭代算法，它在非完全信息游戏中利用了通用的强化学习+搜索，并在一对一无限押注的德州扑克中实现了超人的表现。是Master团队在非完美信息博弈中实现的的一种扑克AI，在德州扑克一对一的有限押注已经取得一定成果。MasterAI于2020年9月战胜了中国的14位最顶级扑克职业选手；

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

 MasterAI 是Master团队在非完美信息博弈中实现的的一种扑克AI，在德州扑克一对一的有限押注（0~100BB）已经取得一定成果，
 MasterAI于2020年9月战胜了中国的14位顶级扑克职业选手；
 在与国内14位顶尖牌手激烈角逐31561手牌后，MasterAI 最终以23,562总计分牌，每百手赢取36.38个大盲的优势取胜。
 MasterAI 基于深度学习，强化学习和博弈论，采用Nash纳什均衡的对战策略，通过大量MC蒙特卡洛采样来计算CFR (虚拟遗憾最小化)的值域或频域作为行动Value，不断探索和选取GTO最优策略，形成智能分析和决策。MasterAI赛事情况如下 ：
 
![640](https://github.com/user-attachments/assets/eeec4ade-bc6b-4359-82ff-0fcb3a72a566)
![640 (1)](https://github.com/user-attachments/assets/d1adb39c-a718-441e-8de6-a1b69d7f356f)

9/1~9/4 首届全明星邀请赛，MasterAI 机器人已战胜顶尖扑克游戏职业高手每百手赢取大盲达到平均36.38的水准，大赢人类职业选手。

MasterAI赛事情况如表 ：人类职业高手巅峰对打表演赛名单

![微信图片_20241030103520](https://github.com/user-attachments/assets/f749ed51-68b6-4ca0-aa2c-693df83e8ece)


Master AI 的训练模型数据以及核心算法代码有偿出售。有兴趣者联系：telegram：@alibaba401

 

- [LeetCode](#leetcode)
    + [1.两数之和](#1两数之和)
    + [2.两数相加](#2两数相加)
    + [3.无重复字符的最长子串](#3无重复字符的最长子串)
    + [5.最长回文子串](#5最长回文子串)
    + [1226.哲学家进餐](#1226哲学家进餐)
    + [820.单词的压缩编码](#820单词的压缩编码)
    + [146.LRU缓存机制(重点)](#146LRU缓存机制重点)
    + [460.LFU缓存(重点)](#460LFU缓存重点)
    + [365.水壶问题](#365水壶问题)
    + [912.排序](#912排序)
    + [885.螺旋矩阵](#885螺旋矩阵)
    + [799.香槟塔](#799香槟塔)
    + [673.最长递增子序列的个数](#673最长递增子序列的个数)
    + [1215.步进数](#1215步进数)
    + [42.接雨水](#42接雨水)
    + [135.分发糖果](#135分发糖果)
    + [96.不同的二叉搜索树](#96不同的二叉搜索树)
    + [351.安卓系统手势解锁](#351安卓系统手势解锁)
    + [1223.掷骰子模拟](#1223掷骰子模拟)
    + [543.二叉树的直径](#543二叉树的直径)
    + [545.判断是否为二叉搜索树和AVL树](#545判断是否为二叉搜索树和avl树)
    + [94.二叉树的前中后序遍历(递归与非递归)](#94二叉树的前中后序遍历递归与非递归)
    + [99.恢复二叉搜索树](#99恢复二叉搜索树)
    + [100.相同的树](#100相同的树)
    + [101.对称的二叉树](#101对称的二叉树)
    + [1143.最长公共子序列](#1143最长公共子序列)
    + [1035.不相交的线(最长公共子序列)](#1035不相交的线最长公共子序列)
    + [758.字符串中的加粗单词](#758字符串中的加粗单词)
    + [107.二叉树的自下向上层次遍历](#107二叉树的自下向上层次遍历)
    + [LCP 3.机器人大冒险](#lcp-3机器人大冒险)
    + [421.数组中两个数的最大异或值](#421数组中两个数的最大异或值)
    + [519.随机翻转矩阵](#519随机翻转矩阵)
    + [1139.最大的以1为边界的正方形](#1139最大的以1为边界的正方形)
    + [1249.移除无效的括号](#1249移除无效的括号)
    + [790.多米诺和托米诺平铺](#790多米诺和托米诺平铺)
    + [4.寻找两个有序数组的中位数](#4寻找两个有序数组的中位数)
    + [6.Z字形变换](#6z字形变换)
    + [1039.多边形三角剖分的最低得分](#1039多边形三角剖分的最低得分)

# LeetCode

### 1.两数之和

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

~~~
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
~~~

创建unordered_map<nums, subscript>，

遍历一遍vector 

若target-nums[i] 存在hashmap中 则输出， 

如不存在 则将{nums[i], i}（数字，下标）加入map

~~~c++
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> mapOfNum;
    for(int i=0; i<nums.size(); i++){
        int element = target - nums[i];
        if(mapOfNum.find(element)!=mapOfNum.end()){
            return {mapOfNum[element], i};
        }
        mapOfNum[nums[i]] = i;
    }
    return {};
}
~~~

### 2.两数相加

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

创建一个节点new ListNode(0)，每一次循环新增一个addNum的节点， 将上一个节点指向新节点， 最终是1需要多加一位new ListNode(1)

~~~c++
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* res = new ListNode(0);
    int addNum = 0;
    int carry = 0;
    ListNode* sup = res;
    int l1val = 0;
    int l2val = 0;
    while(l1 || l2){
        if(l1){
            l1val = l1->val;
            l1 = l1->next;
        }else l1val = 0;
        if(l2){
            l2val = l2->val;
            l2 = l2->next;
        }else l2val = 0;
        addNum = l1val + l2val + carry;
        carry = addNum/10;
        res->next = new ListNode(addNum%10);
        res = res->next;
    }
    if(carry == 1){
        res->next = new ListNode(1);
    }
    return sup->next;
}
~~~



### 3.无重复字符的最长子串

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

**解：**滑动窗口 [i ,j]表示不重复子串，用unordered_map<char, int>存放{字符，下标}。如果map中存在字符，则重置 i = max(i, map[char]) 比较窗口左值和重复字符位置哪个更靠右。 计算maxLength = max(maxLength, j-i+1) ，更新map[char] = j

~~~c++
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> subscript;
    int maxLength = 0;
    for(int i=0,j=0;j<s.length();j++){
        if(subscript.find(s[j])!=subscript.end()){
            i = max(i, subscript[s[j]]+1);
        }
        maxLength = max(maxLength, j-i+1);
        subscript[s[j]] = j;
    }
    return maxLength;
}
~~~



### 5.最长回文子串

```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
```

```
输入: "cbbd"
输出: "bb"
```

中心扩展法，计算每个(i, i)和(i, i+1)的回文长度

[start, end]为回文字符

~~~c++
class Solution {
public:
    string longestPalindrome(string s) {
        if(s.empty())return "";
        int maxLength = 0;
        int eachSubstr = 0;
        for(int i=0;i<s.length();i++){
            eachSubstr = max(lengthOfSubstr(s,i,i), lengthOfSubstr(s,i,i+1));
            if(eachSubstr>maxLength){
                maxLength = eachSubstr;
                res = s.substr(i-(maxLength-1)/2, maxLength);
            }
        }
        return res;
    }
    int lengthOfSubstr(string &s, int start, int end){
        while(start>=0 && end < s.length() && s[start] == s[end]){
            start--;
            end++;
        }
        return end - start - 1;
    }
private:
    string res;
};
~~~



### 1226.哲学家进餐

**三种方法**

1.只允许最多4个人拿起筷子，因此必能保证一个人就餐

2.必须同时能够拿起两双筷子 才能就餐

3.偶数号哲学家 先拿 **左筷子**  奇数号哲学家 先拿 **右筷子**

方法1:

~~~c++
class Semaphore {
public:
    Semaphore(int count = 0) : count_(count) {
    }

    void Set(int count){
        count_ = count;
    }

    void Signal() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_one();
    }

    void Wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        while(count_ <= 0){
            cv_.wait(lock);
        }
        --count_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
};

class DiningPhilosophers {
public:
    DiningPhilosophers() {
        guid.Set(4);
    }

    void wantsToEat(int philosopher,
                    function<void()> pickLeftFork,
                    function<void()> pickRightFork,
                    function<void()> eat,
                    function<void()> putLeftFork,
                    function<void()> putRightFork) {
        int l = philosopher;
        int r = (philosopher+1)%5;
        guid.Wait();        

        lock[l].lock();
        lock[r].lock();
        pickLeftFork();
        pickRightFork();
        eat();
        putRightFork();
        putLeftFork();
        lock[r].unlock();
        lock[l].unlock();

        guid.Signal();
    }
private:
    std::mutex lock[5];
    Semaphore guid;
};
~~~



方法2：

~~~c++
class DiningPhilosophers {
public:
    DiningPhilosophers() {
    }

    void wantsToEat(int philosopher,
                    function<void()> pickLeftFork,
                    function<void()> pickRightFork,
                    function<void()> eat,
                    function<void()> putLeftFork,
                    function<void()> putRightFork) {
        int l = philosopher;
        int r = (philosopher+1)%5;
        guid.lock();        
        lock[l].lock();
        lock[r].lock();
        pickLeftFork();
        pickRightFork();
        guid.unlock();
        eat();
        putRightFork();
        putLeftFork();
        lock[l].unlock();
        lock[r].unlock();
    }
private:
    std::mutex lock[5];
    std::mutex guid;
};
~~~



方法3：

~~~c++
class DiningPhilosophers {
public:
    DiningPhilosophers() {
    }

    void wantsToEat(int philosopher,
                    function<void()> pickLeftFork,
                    function<void()> pickRightFork,
                    function<void()> eat,
                    function<void()> putLeftFork,
                    function<void()> putRightFork) {
        int l = philosopher;
        int r = (philosopher+1)%5;
        if(philosopher%2 == 0){
            lock[r].lock();
            lock[l].lock();
            pickLeftFork();
            pickRightFork();
        }else{
            lock[l].lock();
            lock[r].lock();
            pickLeftFork();
            pickRightFork();
        }

        eat();
        putRightFork();
        putLeftFork();
        lock[l].unlock();
        lock[r].unlock();
    }
private:
    std::mutex lock[5];
};
~~~



### 820.单词的压缩编码

~~~
输入: words = ["time", "me", "bell"]
输出: 10
说明: S = "time#bell#" ， indexes = [0, 2, 5] 。
~~~

字典树解法：

将单词倒序 加入字典树

~~~
		root
		/  \
	   e	l
	  /      \
	 m        l
    /          \
   i            e
  /              \
 t                b
~~~

将每个最长路径的叶子节点 的len+1 (+1是因为要加#)

node->len = words[i].size() + 1;

res = res + len;

TreeNode结构：

~~~c++
struct TreeNode{
    int len;
    map<char, TreeNode*> next;
    TreeNode():len(0){}
};
~~~

map记录 读取的char 应该走的路径 即 下一个TreeNode指针

如果没有 就新建一个TreeNode

~~~c++
int minimumLengthEncoding(vector<string>& words) {
    if(words.empty())return 0;
    TreeNode* root = new TreeNode();
    int res = 0;
    for(int i=0; i<words.size(); i++){
        TreeNode* node = root;
        for(int j=words[i].length()-1; j>=0; j--){
            char c = words[i][j];
            if(node->next.count(c)==0){
                node->next[c] = new TreeNode();
            }
            node = node->next[c];
            //该word已经超过了一个记录的word 因此要删去上次的记录
            if(node->len > 0){
                res -= node->len;
                node->len = 0;
            }
        }
        //只有当该节点是 最远的节点 且没有下一个节点的时候
        if(node->len==0 && node->next.empty()){
            node->len = words[i].size() + 1;
            res += node->len;
        }
    }
    return res;
}
~~~



### 146.LRU缓存机制(重点)

获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。

写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。

**解：**需要一个list存储key 这样可以实现O(1)的 

插入新值：list.push_front();

删除最不常用key: list.pop_back();

因此 最常用在链表头，最不常用在链表尾部，**所以要记得更新操作后将其key移动到链表头部。**

因为 当 key已存在，这时需要更新value值，所以必须知道每个key所指向的list::iterator，需要将其更新至链表头，所以需要unordered_map<int, Value>；

Value定义为一个结构体，需要包含value和list<int>::iterator;

**put操作**：插入值

1.如果已有，则更新：

​	更新map中的value值

​	将相应的list::iterator移至链表头

2.没有，则需要插入：

​	如果map已满: map.size() == capacity

​	则需要先 删除list表尾的key 和 在map中的{key, Value}

​	然后

​	list.push_front(key);	

​	创建一个新的结构体Value(value, list.begin());

​	map中插入{key, Value}

**get操作**：取出value

1.如果map中存在key

​	**记得更新**：将map中对应的iter移至链表头部

​	然后再 return value；

2.不存在 则return -1;

~~~c++
class LRUCache {
public:
    struct Value{
        int value;
        list<int>::iterator iter;
        Value(int v, list<int>::iterator it):value(v), iter(it){}
    };

    LRUCache(int capacity) {
        this->capacity = capacity;
        mHash.reserve(capacity);
    }
    
    int get(int key) {
        auto it = mHash.find(key);
        if(it != mHash.end()){
            mList.splice(mList.begin(), mList, it->second.iter);
            return it->second.value;
        }
        else return -1;
    }
    
    void put(int key, int value) {
        auto it = mHash.find(key);
        if(it != mHash.end()){
            it->second.value = value;
            mList.splice(mList.begin(), mList, it->second.iter);
        }else {
            if(mHash.size() == capacity){
                mHash.erase(mList.back());
                mList.pop_back();
            }
            mList.push_front(key);
            Value v(value, mList.begin());
            mHash.insert({key, v});
        }
    }
private:
    int capacity;
    unordered_map<int, Value> mHash;
    list<int> mList;
};
~~~



### 460.LFU缓存(重点)

设计并实现最不经常使用（LFU）缓存的数据结构。它应该支持以下操作：get 和 put。

~~~c++
struct{
    int value; //value
    int count; //数量
    list<pair<int, list<int>>>::iterator countIt;//对应的count位置 即对应mList位置
    list<int>::iterator keyIt;//对应的key位置
}
unordered_map<int, Val> mHash;
list<pair<int, list<int>>> mList;
int capacity = 0;
~~~

mHash 记录 key 对应的 Val

mList相当于 以count为索引， 记录了同一count中 的key

~~~c++
class LFUCache {
public:
    struct Val{
        int value;
        int count;
        list<pair<int, list<int>>>::iterator countIt;
        list<int>::iterator keyIt;
        Val(int val):value(val), count(1){}
    };
    int capacity = 0;
    unordered_map<int, Val> mHash;
    list<pair<int, list<int>>> mList;

    LFUCache(int capacity) {
        this->capacity = capacity;
    }
    
    int get(int key) {
        auto it = mHash.find(key);
        if(it != mHash.end()){
            update(key, it->second);
            return it->second.value;
        }else return -1;
    }
    
    void put(int key, int value) {
        auto it = mHash.find(key);
        if(it != mHash.end()){
            it->second.value = value;
            update(key, it->second);
        }else if(capacity > 0){
            if(mHash.size() == capacity){
                auto cnt = mList.begin();
                int rmvKey = cnt->second.back();
                cnt->second.pop_back();
                if(cnt->second.empty())mList.erase(cnt);
                mHash.erase(rmvKey);
            }
            auto cnt = mList.begin();
            if(cnt == mList.end() || cnt->first != 1){
                cnt = mList.insert(cnt, {1, list<int>()});
            }
            cnt->second.push_front(key);
            Val v(value);
            v.countIt = cnt;
            v.keyIt = cnt->second.begin();
            mHash.insert({key, v});
        }
    }

    void update(int key, Val &v){
        v.count++;
        auto cnt = v.countIt;
        auto cnt_next = next(cnt);
        if(cnt_next == mList.end() || cnt_next->first != v.count){
            cnt_next = mList.insert(cnt_next, {v.count, list<int>()});
        }
        cnt_next->second.push_front(key);
        cnt->second.erase(v.keyIt);
        if(cnt->second.empty())mList.erase(cnt);
        v.countIt = cnt_next;
        v.keyIt = cnt_next->second.begin();
    }   
};
~~~



### 365.水壶问题

有两个容量分别为 x升 和 y升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？

如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。

你允许：

装满任意一个水壶
清空任意一个水壶
从一个水壶向另外一个水壶倒水，直到装满或者倒空

**解：**利用辗转相除法gcd

如果z<=x+y && z是x,y最大公约数的倍数的话 ，则return true

~~~c++
bool canMeasureWater(int x, int y, int z) {
    return (z==0) || (x+y>=z && z%gcd(x,y)==0);
}
int gcd(int a, int b){
    if(b==0)return a;
    return gcd(b,a%b);
}
~~~

也可以使用深度优先遍历

用pair<int, int>存储 x, y容器的水量

初始为{0, 0}

设置一个visited hash_set<int, int>避免重复

总共6种状态

给x装满：{x, cury}

给y装满: {curx, y}

清空x: {0, cury}

清空y: {curx, 0}

x->y: 需要判断溢出 (curx+cury >y) ? {curx+cury-y, y} : {0, curx+ cury}

y->x: 需要判断溢出 (curx+cury >x) ? {x, cury+curx-x} : {curx+ cury, 0}



### 912.排序

**1.快排**

~~~c++
vector<int> sortArray(vector<int>& nums) {
    quickSort(nums, 0, nums.size()-1);
    return nums;
}
void quickSort(vector<int>& nums, int start, int end){
    if(start >= end)return;
    int random = randomNum(start, end);
    swap(nums[end], nums[random]);
    int small = start-1;
    for(int i=start; i<end; i++){
        if(nums[i] < nums[end]){
            small++;
            if(small < i){
                swap(nums[i], nums[small]);
            }
        }
    }
    small++;
    swap(nums[small], nums[end]);
    quickSort(nums, small+1, end);
    quickSort(nums, start, small-1);
}
int randomNum(int start, int end){
    std::srand(std::time(nullptr));
    return start + std::rand()/((RAND_MAX + 1u)/(end - start + 1));
}
~~~



### 885.螺旋矩阵

在 R 行 C 列的矩阵上，我们从 (r0, c0) 面朝东面开始

**解：**模拟矩阵的行径，如果在矩阵内，则加入res中，当res.size() == R*C时输出

用dr和dc模拟 行走方向

用dk模拟 行走距离

~~~c++
vector<vector<int>> spiralMatrixIII(int R, int C, int r0, int c0) {
    if(R<=0 || C<=0 || r0<0 || r0>=R || c0<0 || c0>=C)return {};
    int dr[] = {0, 1, 0, -1};
    int dc[] = {1, 0, -1, 0};
    vector<vector<int>> res;
    res.push_back({r0,c0});
    if(R*C == 1)return res;
    for(int k=1; k<2*(R+C); k+=2){
        for(int i=0; i<4; i++){ //direction
            int dk = k + i/2; //number of steps
            for(int j=0; j<dk; j++){ //for each step in this direction
                r0 += dr[i];
                c0 += dc[i];
                if(r0>=0 && r0<R && c0>=0 && c0<C){
                    res.push_back({r0,c0});
                    if(res.size()==R*C)return res;
                }
            }
        }
    }
    return {};
}
~~~



### 799.香槟塔

模拟香槟倾倒过程

将mock[0\][0]置为倾倒值

每次将**本杯数值-1**，**再除2**，分别更新下一排的**左杯**和**右杯**值

~~~c++
double champagneTower(int poured, int query_row, int query_glass) {
    double mock[100][100];
    memset(mock, 0, sizeof(mock));
    mock[0][0] = poured;
    for(int i=0;i<query_row;i++){
        for(int j=0;j<=i;j++){
            double count = (mock[i][j] - 1)/2.0;
            if(count>0){
                mock[i+1][j] += count;
                mock[i+1][j+1] += count;
            }
        }
    }
    return min(1.0, mock[query_row][query_glass]);
}
~~~



### 673.最长递增子序列的个数

```
输入: [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]
```

```
输入: [2,2,2,2,2]
输出: 5
解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。
```

**动态规划**

设置count[j]: 以nums[j]为结尾的最长递增子序列**个数**

length[j]:以nums[j]为结尾的最长递增子序列**长度**

从头开始浏览到 j-1 位， 如果nums[i] < nums[j] 

则判断length[i]  >= length[j] 则length[j] = length[i] + 1;相当于将 j 附在以前的i结尾序列后方，因此会 +1

length[i] + 1 == length[j] 则 相当于已经出现过 重复长度的 因此count[j] **+=** count[i];

~~~c++
int findNumberOfLIS(vector<int>& nums) {
    int size = nums.size();
    if(size == 0)return 0;
    int* length = new int[size];
    int* count = new int[size];
    for(int i=0;i<size;i++){
        count[i] = 1;
        length[i] = 1;
    }
    for(int j=1;j<size;j++){
        for(int i=0;i<j;i++){
            if(nums[i] < nums[j]){
                if(length[i] >= length[j]){
                    length[j] = length[i] + 1;
                    count[j] = count[i];
                }else if(length[i]+1 == length[j]){
                    count[j] += count[i];
                }
            }
        }
    }
    int maxLength = 0;
    for(int i=0;i<size;i++){
        if(maxLength < length[i])maxLength = length[i];
    }
    int res = 0;
    for(int i=0;i<size;i++){
        if(length[i] == maxLength)res += count[i];
    }
    return res;

}
~~~



### 1215.步进数

如果一个整数上的每一位数字与其相邻位上的数字的绝对差都是 `1`，那么这个数就是一个「步进数」

给你两个整数，low 和 high，请你找出在 [low, high] 范围内的所有步进数，并返回 排序后 的结果。

```
输入：low = 0, high = 21
输出：[0,1,2,3,4,5,6,7,8,9,10,12,21]
```

**解：**广度优先遍历

先将1,2,3,4,5,6,7,8,9加入队列，再循序加入1*10+2  1\*10+0

即num*10 + (num%10 +/- 1) 模拟步进过程

~~~c++
vector<int> countSteppingNumbers(int low, int high) {
    if(low>high)return {};
    vector<int> res;
    deque<int> dequeOfNum;
    if(low==0)res.push_back(0);
    for(int i=1;i<=9;i++){
        dequeOfNum.push_back(i);
    }
    while(!dequeOfNum.empty()){
        int num = dequeOfNum.front();
        dequeOfNum.pop_front();
        int end = num%10;
        if(num > high)break;
        if(num>=low && num<=high)res.push_back(num);
        if(num < INT_MAX/10){
            if(end != 0)
                dequeOfNum.push_back(10*num + end-1);
            if(end != 9)
                dequeOfNum.push_back(10*num + end+1);
        }      
    }
    return res;
}
~~~



### 42.接雨水

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```
输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```

**解：**使用双指针

left = 0; right = height.size()-1;

判断左指针 和 右指针 哪个值**更小**，雨水存储与**小值**有关

左值更小，则先更新leftmax，再计算ans += leftmax - height[left]

~~~c++
int trap(vector<int>& height) {
    if(height.empty())return 0;
    int left = 0;
    int right = height.size()-1;
    int leftmax = 0;
    int rightmax = 0;
    int ans = 0;
    while(left < right){
        if(height[left] < height[right]){
            height[left] < leftmax ? ans += leftmax - height[left] : leftmax = height[left];
            left++;
        }else {
            height[right] < rightmax ? ans += rightmax - height[right] : rightmax = height[right];
            right--;
        }
    }
    return ans;
}
~~~



### 135.分发糖果

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

每个孩子至少分配到 1 个糖果。
相邻的孩子中，评分高的孩子必须获得更多的糖果。

输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。

输入: [1,2,2]
输出: 4
解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。
第三个孩子只得到 1 颗糖果，这已满足上述两个条件。



**解：**先向右遍历一遍，如果r[i] > r[i-1] 则更新 candies[i] = candies[i-1] + 1;

再从右向左遍历，如果r[i] > r[i+1] 则更新 candies[i] = max(candies[i], candies[i+1]+1); 因为candies[i]可能经过第一次遍历 本来就比candies[i+1]+1还大

~~~c++
int candy(vector<int>& ratings) {
    int length = ratings.size();
    if(length == 0)return 0;
    int* candies = new int[length];
    for(int i=0; i<length; i++){
        candies[i] = 1;
    }
    for(int i=1; i<length; i++){
        if(ratings[i] > ratings[i-1])candies[i] = candies[i-1] + 1;
    }
    for(int i=length-2; i>=0; i--){
        if(ratings[i] > ratings[i+1])
            candies[i] = max(candies[i], candies[i+1] + 1);
    }
    int res = 0;
    for(int i=0; i<length; i++){
        res += candies[i];
    }
    return res;
}
~~~



### 96.不同的二叉搜索树

给定一个整数 *n*，求以 1 ... *n* 为节点组成的二叉搜索树有多少种？

~~~
输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
~~~

$$
{\text{n个节点二叉搜索树个数：}}G(n)=\sum_{i=1}^n F(i,n)\\
{\text{n个节点中，以i为根的二叉搜索树个数：}}F(i,n)=G(i-1)G(n-i)\\
{\text{因此G(n)等于：}}G(n)=\sum_{i=1}^n G(i-1)G(n-i)\\
G(0)=1, G(1)=1, 求G(n).
$$



~~~c++
int numTrees(int n) {
    int* dp = new int[n+1];
    memset(dp, 0, (n+1)*sizeof(int));
    dp[0] = 1;
    dp[1] = 1;
    for(int i=2; i<=n; i++){
        for(int j=1; j<=i; j++){
            dp[i] += dp[j-1]*dp[i-j];
        }
    }
    return dp[n];
}
~~~

**输出所有组合成的二叉搜索树：**

采用递归的方法：

设置start = 1和end = n

得到左子树可能的vector加上 右子树可能的vector，用当前root连接左右子树

[start, i-1]  [i+1, end] 其中 **i** 为root

当start > end是记得返回带有nullptr的vector

~~~c++
vector<TreeNode*> generateTrees(int n) {
    if(n <= 0)return {};
    return coreGenerate(1,n);
}
vector<TreeNode*> coreGenerate(int start, int end){
    vector<TreeNode*> res;
    if(start > end){
        res.push_back(nullptr);
        return res;
    }
    for(int i=start; i<=end; i++){
        vector<TreeNode*> left = coreGenerate(start,i-1);
        vector<TreeNode*> right = coreGenerate(i+1,end);
        for(auto l : left){
            for(auto r : right){
                TreeNode* root = new TreeNode(i);
                root->left = l;
                root->right = r;
                res.push_back(root);
            }
        }
    }
    return res;
}
~~~



### 351.安卓系统手势解锁

1.每一个解锁手势必须至少经过 m 个点、最多经过 n 个点。
2.解锁手势里不能设置经过重复的点。
3.假如手势中有两个点是顺序经过的，那么这两个点的手势轨迹之间是绝对不能跨过任何未被经过的点。
4.经过点的顺序不同则表示为不同的解锁手势。

**解：**回溯法

判断：

~~~c++
if(visited[i*3+j])continue;//已经走过
if(abs(r-i)%2==0 && abs(c-j)%2==0 && !visited[(r+i)/2*3+(c+j)/2])continue;
//判断r-i==2则 跳跃，如果跳跃中间的(r+i)/2,(c+j)/2没有visited，则说明跳空了一格，因此需要continue
~~~

~~~c++
int numberOfPatterns(int m, int n) {
    if(m>n || n<=0)return 0;
    bool* visited = new bool[9];
    memset(visited, 0, 9*sizeof(bool));

    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            path++;
            visited[i*3+j] = true;
            backtrack(i,j,m,n,visited);
            visited[i*3+j] = false;
            path--;
        }
    }
    return res;
}
void backtrack(int r, int c, int &m, int &n, bool* visited){
    if(path>=m && path<=n)res++;
    if(path == n)return;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(visited[i*3+j])continue;
            if(abs(r-i)%2==0 && abs(c-j)%2==0 && !visited[(r+i)/2*3+(c+j)/2])continue;
            path++;
            visited[i*3+j] = true;
            backtrack(i,j,m,n,visited);
            visited[i*3+j] = false;
            path--;

        }
    }
}
~~~



### 1223.掷骰子模拟

有一个骰子模拟器会每次投掷的时候生成一个 1 到 6 的随机数。

不过我们在使用它时有个约束，就是使得投掷骰子时，连续 掷出数字 i 的次数不能超过 rollMax[i]（i 从 1 开始编号）。

现在，给你一个整数数组 rollMax 和一个整数 n，请你来计算掷 n 次骰子可得到的不同点数序列的数量。

~~~
输入：n = 2, rollMax = [1,1,2,2,2,3]
输出：34
解释：我们掷 2 次骰子，如果没有约束的话，共有 6 * 6 = 36 种可能的组合。但是根据 rollMax 数组，数字 1 和 2 最多连续出现一次，所以不会出现序列 (1,1) 和 (2,2)。因此，最终答案是 36-2 = 34。
~~~

**解：**动态规划

~~~
dp[i][j][k]
i:掷的次数
j:当前骰子 值为 [1, 6]
k:当前骰子连续的次数 [1, rollmax[j]]
~~~

~~~
k==1: dp[i][j][k] = dp[i-1][所有非j][所有k, 1-rollmax[非j]]
k!=1: dp[i][j][k] = dp[i-1][j][k-1]
~~~

~~~c++
初始化：
vector<vector<vector<int>>> dp(n, vector<vector<int>>(7, vector<int>(16, 0)));//初始化三维数组 都掷为0，必须都掷为0

dp[1][1-6][1] = 1;//初始化掷一次的情况
~~~

~~~c++
int dieSimulator(int n, vector<int>& rollMax) {
    vector<vector<vector<int>>> dp(n+1, vector<vector<int>>(7, vector<int>(16, 0)));
    for(int i=1; i<7; i++){
        dp[1][i][1] = 1;
    }
    int res = 0;
    for(int i=2; i<=n; i++){
        for(int j=1; j<7; j++){
            for(int k=1; k<=rollMax[j-1]; k++){
                if(k!=1){
                    dp[i][j][k] = dp[i-1][j][k-1];
                }else {
                    for(int z=1; z<7; z++){
                        if(z==j)continue;
                        for(int w=1; w<=rollMax[z-1]; w++){
                            dp[i][j][k] = dp[i][j][k]%1000000007 + dp[i-1][z][w]%1000000007;
                        }
                    }
                }
            }
        }
    }
    for(int i=1; i<7; i++){
        for(int j=1; j<=rollMax[i-1]; j++)
            res = res%1000000007 + dp[n][i][j]%1000000007;
    }
    return res%1000000007;
}
~~~



### 543.二叉树的直径

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。

给定二叉树

          1
         / \
        2   3
       / \     
      4   5    

返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

**解：**通过计算节点的深度，**左深度**加**右深度**正好等于**路径长度**

~~~c++
int ans = 0;
int diameterOfBinaryTree(TreeNode* root) {
    depthOfTree(root);
    return ans;
}
int depthOfTree(TreeNode* root){
    if(root == nullptr)return 0;
    int L = depthOfTree(root->left);
    int R = depthOfTree(root->right);
    ans = max(ans, L+R);
    return max(L, R) + 1;
}
~~~



### 545.判断是否为二叉搜索树和AVL树

是否为二叉搜索树：

~~~c++
bool isValidBST(TreeNode* root) {
    long lower = std::numeric_limits<long>::min();
    long upper = std::numeric_limits<long>::max();
    return coreIsValidBST(root,lower,upper);
}
bool coreIsValidBST(TreeNode* root, long lower, long upper){
    if(root==nullptr)return true;
    long lvar = root->val;
    if(lvar<=lower || lvar>=upper)return false;
    return (coreIsValidBST(root->left,lower,lvar) && coreIsValidBST(root->right, lvar, upper));
}
~~~

中序遍历，如果非递增，则false：

~~~C++
bool isValidBST(TreeNode* root) {
    bool res = true;
    long pre = LONG_MIN;
    inOrder(root, pre, res);
    return res;
}
void inOrder(TreeNode* root, long& pre, bool& res){
    if(root == nullptr)return;
    inOrder(root->left, pre, res);
    if(root->val <= pre)res = false;
    pre = root->val;
    inOrder(root->right, pre, res);
}
~~~

是否为AVL树：

~~~c++
bool isBalanced(TreeNode* root) {
    int depth = 0;
    return depthOfTree(root, depth);
}
bool depthOfTree(TreeNode* root, int &depth){
    if(root == nullptr){
        depth = 0;
        return true;
    }
    int left = 0;
    int right =0;
    if(depthOfTree(root->left, left) && depthOfTree(root->right, right)){
        int diff = abs(left - right);
        if(diff <= 1){
            depth = max(left, right) + 1;
            return true;
        }
    }
    return false;
}
~~~



### 94.二叉树的前中后序遍历(递归与非递归)

前序：

~~~c++
void tree::preOrderRecur(TreeNode* root) {
	if (root == nullptr)return;
	cout << root->val << endl;
	preOrderRecur(root->left);
	preOrderRecur(root->right);
}
~~~

~~~c++
void tree::preOrderLoop(TreeNode* root) {
	if (root == nullptr)return;
	stack<TreeNode*> stackOfNodes;
	TreeNode* p = root;
	while (!stackOfNodes.empty() || p) {
		while (p) {
			cout << p->val << endl;
			stackOfNodes.push(p);
			p = p->left;
		}
		if (!stackOfNodes.empty()) {
			TreeNode* temp = stackOfNodes.top();
			stackOfNodes.pop();
			p = temp->right;
		}
	}
}
~~~

中序：

~~~c++
void tree::inOrderRecur(TreeNode* root) {
	if (root == nullptr)return;
	inOrderRecur(root->left);
	cout << root->val << endl;
	inOrderRecur(root->right);
}
~~~

~~~c++
void tree::inOrderLoop(TreeNode* root) {
	if (root == nullptr)return;
	stack<TreeNode*> stackOfNodes;
	TreeNode* p = root;
	while (!stackOfNodes.empty() || p) {
		while (p) {
			stackOfNodes.push(p);
			p = p->left;
		}
		if (!stackOfNodes.empty()) {
			TreeNode* temp = stackOfNodes.top();
			stackOfNodes.pop();
			cout << temp->val << endl;
			p = temp->right;
		}
	}
}
~~~

后序：

~~~c++
void tree::postOrderRecur(TreeNode* root) {
	if (root == nullptr)return;
	postOrderRecur(root->left);
	postOrderRecur(root->right);
	cout << root->val << endl;
}
~~~

~~~c++
void tree::postOrderLoop(TreeNode* root) {
	if (root == nullptr)return;
	stack<TreeNode*> stackOfNodes;
	TreeNode* cur = root;
	TreeNode* last = nullptr;
	while (cur) {
		stackOfNodes.push(cur);
		cur = cur->left;
	}
	while (!stackOfNodes.empty()) {
		cur = stackOfNodes.top();
		if (cur->right && cur->right != last) {
			cur = cur->right;
			while (cur) {
				stackOfNodes.push(cur);
				cur = cur->left;
			}
		}
		else {
			cout << cur->val << endl;
			last = cur;
			stackOfNodes.pop();
		}
	}
}
~~~



### 99.恢复二叉搜索树

二叉搜索树中的两个节点被错误地交换。

请在不改变其结构的情况下，恢复这棵树。

**解：**二叉搜索树**中序遍历** 应该为 递增序列

因此从序列中 找出 **不符合递增**的两个值 进行交换

用vector<TreeNode*> 存储遍历序列

~~~c++
void recoverTree(TreeNode* root) {
    vector<TreeNode*> vectorOfTree;
    inOrder(root, vectorOfTree);
    int first = -1;
    int end = -1;
    for(int i=0; i<vectorOfTree.size()-1; i++){
        if(vectorOfTree[i]->val > vectorOfTree[i+1]->val){
            if(first == -1)first = i;
            else end = i+1;
        }
    }
    if(end == -1){
        swap(vectorOfTree[first]->val, vectorOfTree[first+1]->val);
    }else {
        swap(vectorOfTree[first]->val, vectorOfTree[end]->val);
    }       
}
void inOrder(TreeNode* root, vector<TreeNode*> &vectorOfTree){
    if(root == nullptr)return;
    inOrder(root->left, vectorOfTree);
    vectorOfTree.push_back(root);
    inOrder(root->right, vectorOfTree);
}
~~~



### 100.相同的树

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

    输入:       1         1
              / \       / \
             2   3     2   3
    输出: true

~~~c++
bool isSameTree(TreeNode* p, TreeNode* q) {
    if(!p && !q)return true;
    if(!p || !q)return false;
    if(p->val == q->val){
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }else return false;
}
~~~



### 101.对称的二叉树

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    	1
       / \
      2   2
     / \ / \
    3  4 4  3

**递归：**

~~~c++
bool isSymmetric(TreeNode* root) {
    if(root == nullptr)return true;
    return isSymmetric(root->left, root->right);
}
bool isSymmetric(TreeNode* left, TreeNode* right){
    if(!left && !right)return true;
    if(!left || !right)return false;
    if(left->val == right->val)
        return isSymmetric(left->left, right->right) && isSymmetric(left->right, right->left);
    else return false;
}
~~~

**迭代：**

~~~c++
bool isSymmetric(TreeNode* root) {
    if(root == nullptr)return true;
    deque<TreeNode*> dequeLeft;
    deque<TreeNode*> dequeRight;
    dequeLeft.push_back(root->left);
    dequeRight.push_back(root->right);
    while(!dequeLeft.empty()){
        TreeNode* left = dequeLeft.front();
        TreeNode* right = dequeRight.front();
        dequeLeft.pop_front();
        dequeRight.pop_front();
        if(!left && !right)continue;
        if(!left || !right)return false;
        if(left->val != right->val)return false;
        dequeLeft.push_back(left->left);
        dequeRight.push_back(right->right);
        dequeLeft.push_back(right->left);
        dequeRight.push_back(left->right);         
    }
    return true;
}
~~~



### 1143.最长公共子序列

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
```

**解：**动态规划：

~~~c++
dp[i][j] 表示 text1 i 和 text2 j 为结尾 最长的公共子序列
if(text1[i] == text2[j])dp[i+1][j+1] = dp[i][j] + 1;
else dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1]);
~~~

 ~~~c++
int longestCommonSubsequence(string text1, string text2) {
    int l1 = text1.length();
    int l2 = text2.length();
    vector<vector<int>> dp(l1+1, vector<int>(l2+1, 0));
    for(int i=0; i<l1; i++){
        for(int j=0; j<l2; j++){
            if(text1[i] == text2[j])
                dp[i+1][j+1] = dp[i][j] + 1;
            else dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1]);
        }
    }
    return dp[l1][l2];
}
 ~~~



### 1035.不相交的线(最长公共子序列)

实则与最长公共子序列相同

~~~c++
int maxUncrossedLines(vector<int>& A, vector<int>& B) {
    int l1 = A.size();
    int l2 = B.size();
    vector<vector<int>> dp(l1+1, vector<int>(l2+1, 0));
    for(int i=0; i<l1; i++){
        for(int j=0; j<l2; j++){
            if(A[i] == B[j])
                dp[i+1][j+1] = dp[i][j] + 1;
            else dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1]);
        }
    }
    return dp[l1][l2];
}
~~~



### 758.字符串中的加粗单词

~~~
给定 words = ["ab", "bc"] 和 S = "aabcd"，
正确："a<b>abc</b>d"。
错误："a<b>a<b>b</b>c</b>d"
~~~

**解：**使用字典树的方法

将每个words中的单词创建字典树

从S[0]开始扫描，字典树中的每一条路径

若得到end>index则说明 有需要加粗的单词

若end == index则说明无需加粗

~~~c++
class Solution {
private:
    struct TrieNode{
        int len;
        vector<TrieNode*> children;
        TrieNode():len(0), children(vector<TrieNode*>(26, nullptr)){}
    };

public:
    string boldWords(vector<string>& words, string S) {
        string res;
        TrieNode* root = new TrieNode();
        for(int i=0; i<words.size(); i++){
            add(root, words[i]);
        }
        int index = 0;
        while(index < S.length()){
            int start = index;
            int end = index + find(root, S, index);
            if(end == index){
                res += S[index];
                index++;
            }else {
                while(index < end){
                    index++;
                    end = max(end, index + find(root, S, index));
                }
                res += "<b>" + S.substr(start, end-start) + "</b>";
            }
        }
        return res;
    }
    void add(TrieNode* root, string &word){
        for(int i=0; i<word.size(); i++){
            if(root->children[word[i]-'a'] == nullptr)
                root->children[word[i]-'a'] = new TrieNode();
            root = root->children[word[i]-'a'];
        }
        root->len = word.size();
    }
    int find(TrieNode* root, string &word, int index){
        int len = 0;
        while(index<word.size() && root->children[word[index]-'a'] != nullptr){
            root = root->children[word[index]-'a'];
            index++;
            len = max(len, root->len);
        }
        return len;
    }
};
~~~



### 107.二叉树的自下向上层次遍历

给定二叉树 `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其自底向上的层次遍历为：

```
[
  [15,7],
  [9,20],
  [3]
]
```

**解：**利用队列和递归(栈)

参数为当前层的deque

生成nextlevel的deque 然后递归nextlevel的deque

知道nextlevel为空，即最后一层的时候，打印这一层。

~~~c++
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        if(root == nullptr)return {};
        deque<TreeNode*> dequeOfTree;
        dequeOfTree.push_back(root);
        printTreeNode(dequeOfTree);
        return res;
    }
    void printTreeNode(deque<TreeNode*> curLevel){
        deque<TreeNode*> nextLevel;
        vector<int> nodes;
        while(!curLevel.empty()){
            TreeNode* temp = curLevel.front();
            curLevel.pop_front();
            nodes.push_back(temp->val);
            if(temp->left)nextLevel.push_back(temp->left);
            if(temp->right)nextLevel.push_back(temp->right);
        }
        if(!nextLevel.empty())printTreeNode(nextLevel);
        res.push_back(nodes);
    }
private:
    vector<vector<int>> res;
};
~~~



### LCP 3.机器人大冒险

力扣团队买了一个可编程机器人，机器人初始位置在原点(0, 0)。小伙伴事先给机器人输入一串指令command，机器人就会无限循环这条指令的步骤进行移动。指令有两种：

U:向y轴正方向移动一格
R:向x轴正方向移动一格

不幸的是，在 xy 平面上还有一些障碍物，他们的坐标用obstacles表示。机器人一旦碰到障碍物就会被损毁。

给定终点坐标(x, y)，返回机器人能否完好地到达终点。如果能，返回true；否则返回false。

**解：**

记录一次command走过的路径path，R次数，U次数

计算(x, y)是否在path中：

计算理应减去重复次数，times = min(x/R次数, y/U次数);

x = x - times * R次数;

y = y - times * U次数;

检查path中是否存在处理后的(x, y)

**Trick：**

可以设置一个long值，将x左移31位，然后和y做&操作，得到唯一值，存入unordered_set中。即前32位是x，后32位是y，相当于唯一的{x, y}。

~~~c++
bool robot(string command, vector<vector<int>>& obstacles, int x, int y) {
    int rightTimes = 0;
    int upTimes = 0;
    set<pair<int, int>> path;
    path.insert({0, 0});
    getPath(command, path, rightTimes, upTimes);
    bool res = isInPath(x, y, path, rightTimes, upTimes);
    if(res){
        for(int i=0; i<obstacles.size(); i++){
            if(obstacles[i][0] > x || obstacles[i][1] > y)continue;
            if(isInPath(obstacles[i][0], obstacles[i][1], path, rightTimes, upTimes)){
                res = false;
                break;
            }
        }
    }
    return res;
}
void getPath(string &command, set<pair<int, int>> &path, int &rightTimes, int &upTimes){
    for(int i=0; i<command.length(); i++){
        if(command[i] == 'U')upTimes++;
        else rightTimes++;
        path.insert({rightTimes, upTimes});
    }
}
bool isInPath(int x, int y, set<pair<int, int>> &path, int &rightTimes, int &upTimes){
    int times = min(x/rightTimes, y/upTimes);
    x = x - times * rightTimes;
    y = y - times * upTimes;
    return (path.count({x, y}) == 1);
}
~~~



### 421.数组中两个数的最大异或值

```
输入: [3, 10, 5, 25, 2, 8]

输出: 28

解释: 最大的结果是 5 ^ 25 = 28
```

**解：**

使用掩码的方式，从最高位，逐步提取前N位，放入set

使用假定的res与set中的num进行**异或运算**，如果结果等于另一个在set中的num，则说明该res 可以 由两个set中的num 异或得到

~~~c++
A ^ B = MAX;
MAX ^ A = B;
MAX ^ B = A;
~~~

~~~c++
int findMaximumXOR(vector<int>& nums) {
    int mask = 0;
    int res = 0;
    for(int i=30; i>=0; i--){
        mask = mask | (1 << i);
        unordered_set<int> setOfNums;
        for(int num : nums){
            setOfNums.insert(num & mask);
        }
        int temp = res | (1 << i);
        for(int num : setOfNums){
            if(setOfNums.count(temp ^ num) == 1){
                res = temp;
                break;
            }
        }
    }
    return res;
}
~~~



### 519.随机翻转矩阵

题中给出一个 n 行 n 列的二维矩阵 (n_rows,n_cols)，且所有值被初始化为 0。要求编写一个 flip 函数，均匀随机的将矩阵中的 0 变为 1，并返回该值的位置下标 [row_id,col_id]；同样编写一个 reset 函数，将所有的值都重新置为 0。尽量最少调用随机函数 Math.random()，并且优化时间和空间复杂度。

**解：**

类似于洗牌策略

用hash_map存放 i 位置的牌 的真实 牌所在位置

每次抽取 [0, n-1] 中的牌，然后使得 选中的 i 指向 末尾 n-1

~~~c++
class Solution {
public:
    Solution(int n_rows, int n_cols) {
        rows = n_rows;
        cols = n_cols;
        rem = rows * cols;
    }
    int getRand(int bound){
        std::random_device rd;
        std::mt19937 gen(rd()); 
        std::uniform_int_distribution<> dis(0, bound);
        return dis(gen);
    }
    
    vector<int> flip() {
        int randNum = getRand(--rem);
        int x = v.count(randNum) ? v[randNum] : v[randNum] = randNum;
        v[randNum] = v.count(rem) ? v[rem] : rem;
        return {x/cols, x%cols};
    }
    
    void reset() {
        v.clear();
        rem = rows * cols;
    }
private:
    int rows = 0;
    int cols = 0;
    int rem = 0;
    unordered_map<int, int> v;
};
~~~

**生成随机数：**

~~~c++
#include<iostream>
#include<random>
//生成 [start, end]
int getRandom(int start, int end) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(start, end);
    return dis(gen);
}
~~~



### 1139.最大的以1为边界的正方形

给你一个由若干 0 和 1 组成的二维网格 grid，请你找出边界全部由 1 组成的最大 正方形 子网格，并返回该子网格中的元素数量。如果不存在，则返回 0。

~~~
输入：grid = [[1,1,1],[1,0,1],[1,1,1]]
输出：9
~~~

**解：**

遍历grid 每当 grid[i][j\] == 1时，查看同行能够往右 延续为1 的step

如果step > 1，则判断以 (i, j)为左上角，是否存在边长为step的正方形

即需要判断：

~~~c++
(i,j)-------------
  |              |
  |              |
  |              |
  |--------------|  //判断四周是否都是1
for(int i=col;i<step+col;i++){
    if(grid[row][i]!=1 || grid[row+step-1][i]!=1)return false;
}
for(int i=row;i<step+row;i++){
    if(grid[i][col+step-1]!=1 || grid[i][col]!=1)return false;
}
~~~



~~~c++
int largest1BorderedSquare(vector<vector<int>>& grid) {
    if(grid.empty())return 0;
    int rows = grid.size();
    int cols = grid[0].size();
    int maxSquare = 0;
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(grid[i][j] == 1){
                int step = 1;
                for(int k=j+1; k<cols; k++){
                    if(grid[i][k]==1)step++;
                    else break;
                }
                while(step>1){
                    if(check(grid,i,j,step))break;
                    step--;
                }
                maxSquare = max(maxSquare, step);
            } 
        }
    }
    return maxSquare*maxSquare;
}
bool check(vector<vector<int>>& grid, int row, int col, int step){
    if(row+step>grid.size() || col+step>grid[0].size())return false;
    for(int i=col;i<step+col;i++){
        if(grid[row][i]!=1 || grid[row+step-1][i]!=1)return false;
    }
    for(int i=row;i<step+row;i++){
        if(grid[i][col+step-1]!=1 || grid[i][col]!=1)return false;
    }
    return true;
}
~~~



### 1249.移除无效的括号

给你一个由 '('、')' 和小写字母组成的字符串 s。

你需要从字符串中删除最少数目的 '(' 或者 ')' （可以删除任意位置的括号)，使得剩下的「括号字符串」有效。

~~~
输入：s = "a)b(c)d"
输出："ab(c)d"
~~~

~~~
输入：s = "))(("
输出：""
解释：空字符串也是有效的
~~~

**解：**

与有效的括号思路相同

创建set<int> 存放错误的下标

创建stack 存储 '(' 的下标，当出现 ')' 时，判断栈是否为空

如果不为空 则 stack.pop() ：抵消前一个 '('

为空，则此 ')' 为错误的 应该加入set中

遍历完后，如果栈非空，则说明栈中所有 '(' 无效，将栈中所有的 下标加入 set中

然后通过set中的下标 切分原有的string

~~~~c++
string minRemoveToMakeValid(string s) {
    string res;
    set<int> mistake;
    stack<int> bracket;
    for(int i=0; i<s.length(); i++){
        if(s[i] == '(')bracket.push(i);
        else if(s[i] == ')'){
            if(!bracket.empty())
                bracket.pop();
            else {
                mistake.insert(i);
            }
        }
    }
    while(!bracket.empty()){
        mistake.insert(bracket.top());
        bracket.pop();
    }
    int start = 0;
    for(int index : mistake){
        res += s.substr(start, index-start);
        start = index + 1;
    }
    res += s.substr(start);
    return res;
}
~~~~



### 790.多米诺和托米诺平铺

有两种形状的瓷砖：一种是 2x1 的多米诺形，另一种是形如 "L" 的托米诺形。两种形状都可以旋转。

~~~
XX  <- 多米诺

XX  <- "L" 托米诺
X
~~~

**解：**

动态规划

计算当前层的下一层 是 00, 01, 10, 11四种情况

~~~c++
int numTilings(int N) {
    long dp[] = {1, 0, 0, 0};
    for(int i=0; i<N; i++){
        long ndp[4];
        ndp[0] = dp[0] + dp[3];
        ndp[1] = dp[0] + dp[2];
        ndp[2] = dp[1] + dp[0];
        ndp[3] = dp[0] + dp[1] + dp[2];
        for(int i=0; i<4; i++){
            dp[i] = ndp[i] % 1000000007;
        }
    }
    return (int)dp[0];
}
~~~



### 4.寻找两个有序数组的中位数

给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

~~~
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
~~~

~~~
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
~~~

**解：**

~~~ 
A[0]...A[i-1]  //  A[i]...A[m-1]
B[0]...B[j-1]  //  B[j]...B[n-1]
保证：
左右相等(m+n为偶数) 此时中位数为 (maxleft + minright)/2
或 左比右多1 (m+n为奇数) 此时中位数为 maxleft
左右相等: i + j == (m + n)/2
左比右大1: i + j == m - i + n - j + 1
j = (m + n + 1)/2 - i; (不影响m + n是偶数的情况)
但要求 n >= m 否则 i == m时 j不能满足 j>=0
n >= m 则能满足 i[0, m] j[0, n]

循环 i = (left + right)/2; j = (m + n + 1)/2 - i;
当 i!=0 && j!=n && nums1[i-1] > nums2[j]    i需要减小 搜索i左侧 right = i - 1;
当 j!=0 && i!=m && nums2[j-1] > nums1[i]    i需要增大 搜索i右侧 left = i + 1;

else:
i==0 A左边全空，maxleft只能是B[j-1]
j==0 B左边全空，maxleft只能是A[j-1]
maxleft = max(A[j-1], B[j-1]);
如果  m + n时奇数，则可以直接输出 maxleft

i==m A右边全空，minright只能是B[j]
j==n B右边全空，minright只能是A[i]
minright = min(A[i], B[j]);
输出 (maxleft + minright)/2

注：A[i]表示 A分界左边有i个数，A[0]则左边0个数，A[m]则左边m个数
所以i取值[0, m]并不是[0, m-1]
~~~

~~~c++
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int m = nums1.size();
    int n = nums2.size();
    if(m > n)return findMedianSortedArrays(nums2, nums1);
    int left = 0;
    int right = m;
    while(left <= right){
        int i = left + ((right - left) >> 1);
        int j = ((m + n + 1)>>1) - i;
        if(i!=0 && j!=n && nums1[i-1] > nums2[j])
            right = i - 1;
        else if(j!=0 && i!=m && nums2[j-1] > nums1[i])
            left = i + 1;
        else {
            int maxleft = 0;
            if(i == 0)maxleft = nums2[j-1];
            else if(j == 0)maxleft = nums1[i-1];
            else maxleft = max(nums1[i-1], nums2[j-1]);
            if((m+n)&1 == 1)return maxleft;

            int minright = 0;
            if(i == m)minright = nums2[j];
            else if(j == n)minright = nums1[i];
            else minright = min(nums1[i], nums2[j]);
            return double(maxleft + minright) / 2;
        }
    }
    return 0;
}
~~~



### 6.Z字形变换

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "LEETCODEISHIRING" 行数为 3 时，排列如下：

~~~
L   C   I   R
E T O E S I I G
E   D   H   N
~~~

按行输出："LCIRETOESIIGEDHN"

**解：**

利用数学的方法，推断出来每次输出的下标index

从index出发点为 [0, rows-1]

~~~
i = [0, rows - 1]
j = rows - i; [rows, 1]
如果是第一行 则只走 2*(rows - 1)
如果是最后一行 则只走 2*(rows - 1)
其余行 先走 2*(j-1) 再走 2*(rows - j) 设定一个cur=0; cur = 1 - cur;实现交替
~~~

~~~c++
string convert(string s, int numRows) {
    int length = s.length();
    if(numRows == 1)return s;
    string res;
    for(int i=0; i<numRows; i++){
        int cur = 0;
        int j = numRows - i;
        int index = i;
        while(index < length){
            res += s[index];
            if(j == numRows)index += 2*(j-1);
            else if(j == 1)index += 2*(numRows-j);
            else {
                index += cur==0 ? 2*(j-1) : 2*(numRows-j);
                cur = 1 - cur;
            }            
        }
    }
    return res;
}
~~~

**方法2：**

创建rows个 string 模拟Z字形行走，不断输出到对应行中

~~~c++
string convert(string s, int numRows) {
    if (numRows == 1) return s;
    vector<string> rows(min(numRows, int(s.size())));
    int curRow = 0;
    bool goingDown = false;
    for (char c : s) {
        rows[curRow] += c;
        if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
        curRow += goingDown ? 1 : -1; //改变方向
    }
    string ret;
    for (string row : rows) ret += row;
    return ret;
}
~~~



### 1039.多边形三角剖分的最低得分

给定 N，想象一个凸 N 边多边形，其顶点按顺时针顺序依次标记为 A[0], A[i], ..., A[N-1]。

假设您将多边形剖分为 N-2 个三角形。对于每个三角形，该三角形的值是顶点标记的乘积，三角剖分的分数是进行三角剖分后所有 N-2 个三角形的值之和。

~~~
3 - 7   3 - 7
| / |   | \ |
5 - 4   5 - 4

输入：[3,7,4,5]
输出：144
解释：有两种三角剖分，可能得分分别为：3*7*5 + 4*5*7 = 245，或 3*4*5 + 3*4*7 = 144。最低分数为 144
~~~

**解：**

动态规划

~~~
dp[i][j]表示 从i 到 j构成的多边形 三角剖分 后的最小值
结果应输出 dp[0][N-1]
初始化所有的dp[i][j]为 1000000 因为求最小值
1.初始化dp[i][(i+1)%N] = 0
2.len 从2到N-1 开始循环
循序得到 首节点是i到N-1的 n边形 的 结果
~~~

~~~c++
int minScoreTriangulation(vector<int>& A) {
    int N = A.size();
    if(N < 3)return 0;
    vector<vector<int>> dp(N, vector<int>(N, 1000000));
    for(int i=0; i<N; i++){
        dp[i][(i+1)%N] = 0;
    }
    for(int len=2; len<N; len++){
        for(int i=0; i<N; i++){
            int j = (i+len)%N;
            for(int k=(i+1)%N; k!=j; k=(k+1)%N){
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + A[i]*A[j]*A[k]);
            }
        }
    }
    return dp[0][N-1];
}
~~~


- [LeetCode](#leetcode)
    + [1.两数之和](#1两数之和)
    + [2.两数相加](#2两数相加)
    + [3.无重复字符的最长子串](#3无重复字符的最长子串)
    + [5.最长回文子串](#5最长回文子串)
    + [1226.哲学家进餐](#1226哲学家进餐)
    + [820.单词的压缩编码](#820单词的压缩编码)
    + [146.LRU缓存机制(重点)](#146LRU缓存机制(重点))
    + [460.LFU缓存(重点)](#460LFU缓存(重点))
    + [365.水壶问题](#365水壶问题)
    + [912.排序](#912排序)
    + [885.螺旋矩阵](#885螺旋矩阵)
    + [799.香槟塔](#799香槟塔)
    + [673.最长递增子序列的个数](#673最长递增子序列的个数)
    + [1215.步进数](#1215步进数)
    + [42.接雨水](#42接雨水)
    + [135.分发糖果](#135分发糖果)
    + [96.不同的二叉搜索树](#96不同的二叉搜索树)


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



### 2.两数相加

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

创建一个节点new ListNode(0)，每一次循环新增一个addnum的节点， 将上一个节点指向新节点， 最终是1需要多加一位new ListNode(1)

~~~c++
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
    carry = addNum/10;// 用除10计算进位carry
    res->next = new ListNode(addNum%10); //用%10计算实际值
    res = res->next;
}
if(carry == 1){ //记得最后一位有可能进位为1
    res->next = new ListNode(1);
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
if(len> end-start+1){
  start = i - (len-1)/2;
  end = i + len/2;
}
s.substr(start, end-start+1) //注意第二个参数是字符数量而不是右边界
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
    memset(length, 0, size*sizeof(int));
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


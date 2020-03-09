- [剑指Offer](#--offer)
    + [13.机器人的运动范围](#13机器人的运动范围)
    + [21.调整数组顺序使奇数位于偶数前面](#21---------------)
    + [22.链表中倒数第K个节点](#22------k---)
    + [23.链表中环的入口节点](#23---------)
    + [24.反转链表](#24----)
    + [25.合并两个排序的链表](#25---------)
    + [26.树的子结构](#26-----)
    + [27.二叉树的镜像](#27------)
    + [28.对称的二叉树](#28------)
    + [29.顺时针打印矩阵](#29-------)
    + [30.包含min(max)函数的栈](#30--min-max-----)
    + [31.栈的压入和弹出序列](#31---------)
    + [33.二叉搜索树的后序遍历序列](#33------------)
    + [34.二叉数中和为某一值的路径](#34------------)
    + [35.复杂链表的复制](#35-------)
    + [36.二叉搜索树与双向链表](#36----------)
    + [37.序列化二叉树](#37------)
    + [38.字符串的排列](#38------)
    + [39.数组中出现次数超过一半的数字](#39--------------)
    + [40.最小的K个数](#40---k--)
    + [41.数据流中的中位数](#41--------)
    + [42.连续子数组的最大和](#42---------)
    + [43.1-整数n中1出现的次数](#431---n-1-----)
    + [44.数字序列中某一位的数字](#44-----------)
    + [45.把数组排成最小的数](#45---------)
    + [46.把数字翻译成字符串](#46---------)
    + [47.礼物的最大价值](#47-------)
    + [48.最长不含重复字符的子字符串](#48-------------)
    + [49.丑数](#49--)


# 剑指Offer

### 13.机器人的运动范围

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

**解：**设置visited数组记录已经到达的值，避免重复。

深度优先搜索：

~~~c++
int movingCount(int m, int n, int k) {
    if(m<=0 || n<=0)return 0;
    bool* visited = new bool[m*n];
    memset(visited, 0, m*n*sizeof(bool));
    return coreMovingCount(m, n, 0, 0, k, visited);
}
int coreMovingCount(int &rows, int &cols, int row, int col, int k, bool* visited){
    if(row<0 || row>=rows || col<0 || col>=cols || visited[row*cols+col] || (checkDigit(row, col)>k))return 0;
    visited[row*cols+col] = true;
    return 1 + coreMovingCount(rows,cols,row+1,col,k,visited)
        + coreMovingCount(rows,cols,row-1,col,k,visited)
        + coreMovingCount(rows,cols,row,col+1,k,visited)
        + coreMovingCount(rows,cols,row,col-1,k,visited);

}
int checkDigit(int num1, int num2){
    int res = 0;
    while(num1){
        res += num1%10;
        num1 = num1/10;
    }
    while(num2){
        res += num2%10;
        num2 = num2/10;
    }
    return res;
}
~~~

~~~c++
vector<vector<int>> visited(m,vector<int>(n,0));
~~~



### 21.调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

方法1.

设置两个指针，一个在最前，一个在最后

每当前指针指向偶数，后指针指向奇数时 swap

~~~c++
while(begin<end && (begin&1)==0) //偶数
	begin++;
while(begin<end && (end&1)!=0) //奇数
	end--;
if(begin<end)swap(nums[begin], nums[end]);
~~~

方法2.

partition函数，快排思路，start位是奇数位，start+1位偶数位

~~~c++
int start = -1;
for(int i=0;i<end;i++){
	if(nums[i]&1 ==1){//奇数
		start++;
		if(i>start)swap(nums[i], nums[start]);
	}
}
~~~



### 22.链表中倒数第K个节点

双指针，1->2->3->4->5 若求倒数第二个 即一个指针指向1 一个指针指向2

~~~c++
while(n2->next){
	n1 = n1->next;
	n2 = n2->next;
}
~~~

注意：记得判断k>len 则返回nullptr



### 23.链表中环的入口节点

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null

1.双指针，一个指针走1步 一个指针走2步， 得到重合位置

2.从重合位置 循环 再次到达重合位置，计算环中节点数count

3.meet节点从head节点向后count次next，然后meet==head时 即为入口节点



### 24.反转链表

输入一个链表的头节点，反转该链表并输出反转后链表的头节点

三个指针

~~~c++
ListNode* reverseList(ListNode* head) {
        ListNode* reverseHead = nullptr;
        ListNode* pNode = head;
        ListNode* prev = nullptr;
        ListNode* pNext = nullptr;
        while(pNode){
            pNext = pNode->next;
            if(pNext==nullptr)reverseHead = pNode;

            pNode->next = prev;
            prev = pNode;
            pNode = pNext;
        }
        return reverseHead;
    }
~~~



### 25.合并两个排序的链表

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

~~~c++
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(!l1)return l2;
        if(!l2)return l1;
        ListNode* pHead = nullptr;
        if(l1->val > l2->val){
            pHead = l2;
            pHead->next = mergeTwoLists(l1, l2->next); //递归
        }else {
            pHead = l1;
            pHead->next = mergeTwoLists(l1->next, l2);
        }
        return pHead;
    }
~~~



### 26.树的子结构

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

例如:
给定的树 A:

```
   	3  
   / \  
  4   5 
 / \ 
1   2
```

给定的树 B：

```
   4  
  / 
 1
```

返回 true。

**解：** 需要两个函数，

1.一个函数判断 A节点 是否是 B树的根节点

如果是  则递归其子树是否和B都相等。

不是 则判断A->left 或 A->right是否是B的根节点 

2.第二个函数，递归判断A->val 是否 等于 B->val 

是 则判断 (A->left, B->left) && (A->right, B->right)

不是 则 return false

~~~c++
bool isSubStructure(TreeNode* A, TreeNode* B) {
	if(!A || !B)return false;
	return coreSubStructure(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B);
}
bool coreSubStructure(TreeNode* A, TreeNode* B){
	if(!B)return true;
	if(!A)return false;
	if(A->val == B->val){
		return coreSubStructure(A->left, B->left) && coreSubStructure(A->right, B->right);
	}else return false;
}
~~~



### 27.二叉树的镜像

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

     	 4
       /   \
      2     7
     / \   / \
    1   3 6   9

 镜像输出：

     	 4
       /   \
      7     2
     / \   / \
    9   6 3   1

**解：** 新建root->val 节点，left指向root->right， right指向->root->left即完成镜像操作

  ~~~c++
TreeNode* mirrorTree(TreeNode* root) {
    if(root == nullptr)return nullptr;

    TreeNode* res = new TreeNode(root->val);
    res->left = mirrorTree(root->right);
    res->right = mirrorTree(root->left);
    return res;
}
  ~~~



### 28.对称的二叉树

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

        1
       / \
      2   2
     / \ / \
    3  4 4  3

  把2 ，2 放进函数判断 (2->left, 2->right) && (2->right, 2->left)

​												(3, 3)                             (4, 4)

~~~c++
bool isSymmetric(TreeNode* root) {
    if(root==nullptr)return true;
    return coreIsSymmetric(root->left, root->right);
}
bool coreIsSymmetric(TreeNode* rootL, TreeNode* rootR){
    if(!rootL && !rootR)return true;
    if(!rootL || !rootR)return false;
    if(rootL->val == rootR->val)
        return coreIsSymmetric(rootL->left, rootR->right)
        && coreIsSymmetric(rootL->right, rootR->left);
    return false;
}
~~~



### 29.顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

 

示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]

~~~
1,2,3
4,5,6
7,8,9
~~~

输出：[1,2,3,6,9,8,7,4,5]

start=0  while(rows>2\*start && cols>2\*start) 进入循环

(start, start)为打印一圈的起始位置(左上角)坐标

终止X轴 即 列数 endX  = cols - 1 - start;

终止Y轴 即 行数 endY = rows -1 - start;

```
------>
|	  |
|	  |
<-----|
```

1.无论如何 都将打印一行

2.当 列数 大于2时 ，即endY>start 时 打印最后一列

3.当 行数大于2 且 列数大于2时 即 endY>start && endX>start时，打印最下方一行

4.当 行数大于3 且 列数大于2时 即 endY>start+1 && endX>start时，打印 最左列

~~~c++
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    rows = matrix.size();
    if(rows == 0)return {};
    cols = matrix[0].size();
    int start = 0;
    while(rows>2*start && cols>2*start){
        printNum(matrix, start);
        start++;
    }
    return res;
}
void printNum(vector<vector<int>>& matrix, int start){
    int endX = cols - 1 - start;
    int endY = rows - 1 - start;
    for(int i=start; i<=endX; i++){
        res.push_back(matrix[start][i]);
    }
    if(endY>start){
        for(int i=start+1; i<=endY; i++){
            res.push_back(matrix[i][endX]);
        }
    }
    if(endY>start && endX>start){
        for(int i=endX-1; i>=start; i--){
            res.push_back(matrix[endY][i]);
        }
    }
    if(endX>start && endY>start+1){
        for(int i=endY-1; i>=start+1; i--){
            res.push_back(matrix[i][start]);
        }
    }
}
~~~



### 30.包含min(max)函数的栈

使用两个栈

一个栈正常存储数据

另一个栈 只存最小的值：

每次与minstack.top()比较   x<top 则存入x  否则 继续存一次top

~~~c++
void push(int x) {
    normalStack.push(x);
    if(minStack.empty())minStack.push(x);
    else if(x < minStack.top())minStack.push(x);
    else minStack.push(minStack.top());
}

void pop() {
    normalStack.pop();
    minStack.pop();
}

int top() {
    return normalStack.top();
}

int min() {
    return minStack.top();
}
~~~



### 31.栈的压入和弹出序列

新建辅助栈 stack<int>

**解：**while(pop还有数据): while(栈空 或者 栈顶和pop[index]**不相等**)则持续入栈

当栈顶和pop[index]**相等**时，则popindex++，pop();

~~~c++
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    stack<int> simulate;
    int pushIndex = 0;
    int popIndex = 0;
    while(popIndex < popped.size()){
        while(pushIndex<pushed.size() && (simulate.empty() || simulate.top()!=popped[popIndex])){
            simulate.push(pushed[pushIndex]);
            pushIndex++;
        }
        if(simulate.top()!=popped[popIndex])break;
        simulate.pop();
        popIndex++;
    }
    if(simulate.empty() && popIndex==popped.size())
        return true;
    else return false;
}
~~~



32.从上打下打印二叉树(二叉树层序遍历)

**典型的广度优先遍历 使用队列**

创建deque，将root加入deque

循环 输出deque.front()的值 然后将deque.front()左右节点加入deque中

~~~c++
vector<int> levelOrder(TreeNode* root) {
    if(root==nullptr)return {};
    vector<int> res;
    deque<TreeNode*> dequeOfTree;
    dequeOfTree.push_back(root);
    while(!dequeOfTree.empty()){
        TreeNode* temp = dequeOfTree.front();
        dequeOfTree.pop_front();
        res.push_back(temp->val);
        if(temp->left)dequeOfTree.push_back(temp->left);
        if(temp->right)dequeOfTree.push_back(temp->right);
    }
    return res;
}
~~~

**逐层输出二叉树**

同样使用 deque<TreeNode*>

设置初始 toBePrint = 1； nextLevel = 0；

toBePrint为 需要打印的数量

nextLevel 记录下一层需要打印的数量

当toBePrint==0时

~~~c++
toBePrint = nextLevel;
nextLevel = 0;
~~~

**之字形打印二叉树**

~~~
	  3                     结果[3],[20,9],[5,6,15,7]
    /   \
   9     20
  / \   /  \
 5   6 15   7               
~~~

建立两个堆栈，stack<TreeNode*> stackOfTree[2];

设置current == 0； 将root加入stackOfTree[0]中

~~~
stack0:3 7 15 6 5     右左
stack1;9 20           左右
~~~

每当current == 0; 应该 先左后右 插入 stack[1-current] //1

每当current == 1; 应该 先右后左 插入 stack[1-current] //0

~~~c++
vector<vector<int>> levelOrder(TreeNode* root) {
    if(root == nullptr)return {};
    stack<TreeNode*> stackOfTree[2];
    int current = 0;
    stackOfTree[0].push(root);
    TreeNode* temp = nullptr;
    vector<vector<int>> res;
    vector<int> eachRow;
    while(!stackOfTree[current].empty()){
        temp = stackOfTree[current].top();
        stackOfTree[current].pop();
        eachRow.push_back(temp->val);
        if(current==0){
            if(temp->left)stackOfTree[1-current].push(temp->left);
            if(temp->right)stackOfTree[1-current].push(temp->right);
        }else {
            if(temp->right)stackOfTree[1-current].push(temp->right);
            if(temp->left)stackOfTree[1-current].push(temp->left);
        }
        if(stackOfTree[current].empty()){
            res.push_back(eachRow);
            eachRow.clear();
            current = 1 - current;
        }  
    }
    return res;
}
~~~



### 33.二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。

**解：**输入的遍历数组 最后一项为根节点

因此找出根节点 找出所有**小于根节点的数** 和 所有**大于根节点的数**

分别递归判断是否为后序遍历序列

**判断：** 计算mid值，如果[mid, end-1]仍有小于根节点的值 则return false

mid为 左子树 与 右子树 划分点

~~~c++
bool verifyPostorder(vector<int>& postorder) {
    return coreVerify(postorder, 0, postorder.size()-1);
}
bool coreVerify(vector<int>& postorder, int start, int end){
    if(start >= end)return true;
    int mid = start;
    while(postorder[mid]<postorder[end])++mid;
    for(int i=mid; i<end; i++){
        if(postorder[i]<postorder[end])return false;
    }
    return coreVerify(postorder, start, mid-1)
        && coreVerify(postorder, mid, end-1);
}
~~~



### 34.二叉数中和为某一值的路径

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

**解：**此题 必须是 从 **根节点 **到 **叶节点** 的完整路径

典型的**深度优先遍历**，得到所有的路径

当某节点 是 **叶子节点** 且 **值和sum相同** 时 则输出一组答案 然后继续遍历

~~~c++
void dfs(TreeNode* root, int sum){
    temp.push_back(root->val);
    //判断是否为叶子节点,且满足sum
    if(!root->left && !root->right && root->val == sum){
        res.push_back(temp);
    }
    if(root->left)dfs(root->left, sum-root->val);
    if(root->right)dfs(root->right, sum-root->val);

    temp.pop_back();
}
//vector<vector<int>> res; 
//vector<int> temp;
~~~



### 35.复杂链表的复制

使用unordered_map<oldNode\*, newNode\*>记录新建的Node和原Node的关系

先遍历一遍链表，得到完整的map

再遍历一遍:

~~~c++
while(pNode){
    newNode = mapOfNodes[pNode];
    if(pNode->next)newNode->next = mapOfNodes[pNode->next];
    if(pNode->random)newNode->random = mapOfNodes[pNode->random];
    pNode = pNode->next;
}
~~~



### 36.二叉搜索树与双向链表

**解：**设置Node* lastNode; 用于表示已经得到的链表的最后一个节点

中序遍历，左根右，当遍历到根节点时，使得root和lastNode互相指向对方

更新lastNode = root

一开始设置lastNode = nullptr， 保证链表头节点指向nullptr

主函数：遍历出最左节点得到head,然后让head和得到的lastNode相互指向

~~~c++
Node* treeToDoublyList(Node* root) {
    if(root==nullptr)return nullptr;
    Node* lastNode = nullptr;
    inOrder(root, &lastNode);

    while(root->left){
        root = root->left;
    }
    root->left = lastNode;
    lastNode->right = root;
    return root;
}
void inOrder(Node* root, Node** lastNode){
    if(root->left)
        inOrder(root->left, lastNode);
    root->left = *lastNode;
    if((*lastNode)!=nullptr)
        (*lastNode)->right = root;
    *lastNode = root;
    if(root->right)
        inOrder(root->right, lastNode);
}
~~~



### 37.序列化二叉树

    将以下二叉树：
    	1
       / \
      2   3
         / \
        4   5
    
    序列化为 "[1,2,3,null,null,4,5]"

采用**广度优先遍历**

**序列化：**

deque可以push_back(nullptr)

得到deque.front()判断是否为nullptr，

是，则res += "#,"

不是， 则res += to_string(val) + ','  再将 left, right都进队列

**反序列化：**

使用一个takeNum 函数 采用int &start进行 计步器功能

返回 TreeNode(val) || 若为'#' 返回nullptr

~~~c++
string serialize(TreeNode* root) {
    string res;
    deque<TreeNode*> dequeOfTree;
    dequeOfTree.push_back(root);
    while(!dequeOfTree.empty()){
        TreeNode* temp = dequeOfTree.front();
        dequeOfTree.pop_front();
        if(temp){   
            res += to_string(temp->val) + ',';
            dequeOfTree.push_back(temp->left);
            dequeOfTree.push_back(temp->right);
        }else {
            res += "#,";
        }
    }
    return res;
}

TreeNode* deserialize(string data) {
    int start = 0;
    if(data[start]=='#')return nullptr;
    TreeNode* root = takeNum(data, start);
    deque<TreeNode*> dequeOfTree;
    dequeOfTree.push_back(root);
    while(!dequeOfTree.empty()){
        TreeNode* temp = dequeOfTree.front();
        dequeOfTree.pop_front();
        temp->left = takeNum(data, start);
        if(temp->left)dequeOfTree.push_back(temp->left);
        temp->right = takeNum(data, start);
        if(temp->right)dequeOfTree.push_back(temp->right);
    }
    return root;
}

TreeNode* takeNum(string &data, int &start){
    if(data[start]=='#'){
        start = start+2;
        return nullptr;
    }
    int sig = 0;
    int num = 0;
    if(data[start]=='-'){
        sig = 1;
        start++;
    }
    while(data[start]!=','){
        num = num*10 + data[start]-'0';
        start++;
    }
    if(sig==1)num = -num;
    start++;
    TreeNode* res = new TreeNode(num);
    return res;
}
~~~



### 38.字符串的排列

```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

如果有重复 需要去重 **需要先把输入排序** **需要先把输入排序** **需要先把输入排序！！！！！！！！**小trick

排序完 执行普通的递归即可，不断的交换位置，得到全排列

~~~c++
vector<string> permutation(string s) {
    sort(s.begin(),s.end());//排列 ！！排列 ！！排列 ！！
    corePermutation(s, 0);
    return res;
}
void corePermutation(string s, int index){
    if(index == s.length()){
        res.push_back(s);
        return;
    }
    for(int i=index;i<s.length();i++){
        if(i!=index && s[i] == s[index])continue;//去重！！！！！
        swap(s[i], s[index]);
        corePermutation(s, index+1);
        //swap(s[i], s[index]);
    }
}
~~~



### 39.数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

```
输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
```

**解：** 采用抵消的思路，若当前数字与前一个数字不相等，则抵消

**两两抵消**，最后剩下的一定是出现次数超过一半的数字

**记得遍历一遍输入数组，检查count是否大于一半** 

~~~c++
int majorityElement(vector<int>& nums) {
    if(nums.empty())return -1;
    int count = 1;
    int resNum = nums[0];
    int mid = nums.size() >> 2;
    for(int i=1;i<nums.size();i++){
        if(nums[i] == resNum){
            count++;
        }else {
            count--;
            if(count<0){
                resNum = nums[i];
                count = 1;
            }
        }
    }
    if(count>0){
        count = 0;
        for(int i=0;i<nums.size();i++){
            if(nums[i] == resNum)count++;
        }
    }
    return (count > mid) ? resNum : -1;
}
~~~



### 40.最小的K个数

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

**解：**利用patition函数 知道mid = k-1为止

~~~c++
int start = 0;
int end = arr.size() - 1;
int mid = partition(arr, start, end);
while(mid != k-1){
    if(mid<k)
        start = mid + 1;
    else end = mid - 1;
    mid = partition(arr, start, end);
}
~~~

**partition函数：**

~~~c++
int partition(vector<int> &arr, int start, int end){
    int rand = randomNum(start, end);
    swap(arr[end], arr[rand]);
    int index = start - 1;
    for(int i=start;i<end;i++){
        if(arr[i]<arr[end]){
            index++;
            if(i!=index){
                swap(arr[index], arr[i]);
            }
        }
    }
    index++;
    swap(arr[index], arr[end]);
    return index;
}
~~~

**randomNum函数：**注意括号！

~~~c++
int randomNum(int start, int end){
    std::srand(std::time(nullptr));
    return start + std::rand()/((RAND_MAX + 1u)/(end - start + 1));
}
~~~

**方法2：**使用multiset

建立std::multiset<int, greater<int>> setOfNums;

当set.size()小于K时 插入

else 比较 begin处的值和nums[i]

如果begin处较大，则应该舍弃begin值，再将nums[i]插入set

~~~c++
vector<int> getLeastNumbers(vector<int>& arr, int k) {
    if(arr.empty() || k<=0 || k>arr.size())return {};
    multiset<int, greater<int>> setOfNums;
    vector<int> res;
    for(int i=0;i<arr.size();i++){
        if(setOfNums.size()<k){
            setOfNums.insert(arr[i]);
        }else {
            auto iter = setOfNums.begin();
            if(*iter > arr[i]){
                setOfNums.erase(iter);
                setOfNums.insert(arr[i]);
            }
        }
    }
    for(auto iter = setOfNums.begin();iter!=setOfNums.end();iter++){
        res.push_back(*iter);
    }
    return res;
}
~~~



### 41.数据流中的中位数

利用最大堆和最小堆

数据结构满足    **[最大堆] 中位数 [最小堆]**

若总size为**奇数** 则默认为 最大堆的顶点为 中位数

若总size为**偶数** 则 最大堆和最小堆的顶点 **相加除2** 为中位数

**具体流程：**

1.偶数时 默认加入最大堆，需要判断加入num 是否 **大于最小堆顶点**

如果大于，则不能直接加入最大堆，需要将其与最小堆 的最小值加入最大堆中

因为 直接加入 则会引发 最大堆中的值 大于 最小堆中的值 的情况， 出现错误

2.奇数时 默认加入最小堆，需要判断加入num 是否 **小于最大堆顶点**

~~~c++
void addNum(int num) {
    if(((maxHeap.size() + minHeap.size())&1) == 0){
        if(minHeap.size()>0 && num > minHeap[0]){
            minHeap.push_back(num);
            push_heap(minHeap.begin(),minHeap.end(),std::greater<int>());
            num = minHeap[0];
            pop_heap(minHeap.begin(),minHeap.end(),std::greater<>{});
            minHeap.pop_back();
        }
        maxHeap.push_back(num);
        push_heap(maxHeap.begin(), maxHeap.end());
    }else {
        if(maxHeap.size()>0 && num < maxHeap[0]){
            maxHeap.push_back(num);
            push_heap(maxHeap.begin(), maxHeap.end());
            num = maxHeap[0];
            pop_heap(maxHeap.begin(), maxHeap.end());
            maxHeap.pop_back();
        }
        minHeap.push_back(num);
        push_heap(minHeap.begin(), minHeap.end(), std::greater<>{});
    }
}

double findMedian() {
    if(((maxHeap.size() + minHeap.size())&1) == 0){
        return (double)(minHeap[0] + maxHeap[0])/2;
    }else return maxHeap[0];
}
~~~

**关于堆的语法：**

~~~c++
vector<int> v;
push_heap(v.begin(), v.end()); //创建最大堆
push_heap(v.begin(), v.end(), std::greater<int>()); //创建最小堆1
push_heap(v.begin(), v.end(), std::greater<>{}); //创建最小堆2

pop_heap(v.begin(), v.end()); //移除堆顶元素 至 末尾
v.pop_back(); //移除堆顶元素后，使用pop_back()即可移除该元素

//更新步骤:
v.push_back(num);
push_heap(v.begin(), v.end());
pop_heap(v.begin(), v.end());
v.pop_back();
~~~



### 42.连续子数组的最大和

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**解：** 记录当前currentNum 和 maxNum

如果currentNum >0 则currentNum 加上num[i] 

否则 currentNum = num[i]

~~~c++
int maxSubArray(vector<int>& nums) {
    if(nums.empty())return -1;
    int current = nums[0];
    int res = nums[0];
    for(int i=1;i<nums.size();i++){
        if(current > 0)
            current = current + nums[i];
        else current = nums[i];
        res = max(res, current);
    }
    return res;
}
~~~



### 43.1-整数n中1出现的次数

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

~~~c++
先将n转换成string num = to_string(n)；

nextNum = num.substr(1);

length = num.size();

first为第一位num[0] - ‘0‘；

firstDigit: 如果 first > 1: pow(10, length-1);
                first == 1: stoi(nextNum) + 1;
                first == 0: 0;
otherDidit: first * (length-1) * pow(10, length-2);
nextDigit: 函数递归(nextNum);

临街判断:num.size()==1 :
first >=1 return 1;
first ==0 return 0;
~~~

**具体代码：**

~~~c++
int countDigitOne(int n) {
    if(n<1)return 0;
    string num = to_string(n);
    return numOfOne(num);
}
int numOfOne(string num){
    int firstNum = num[0] - '0';
    if(num.size()==1){
        if(firstNum >= 1)return 1;
        else return 0;
    }
    string nextNum = num.substr(1);
    int firstDigit = 0;
    if(firstNum>1)firstDigit = pow(10,num.size()-1);
    else if(firstNum==1)firstDigit = stoi(nextNum) + 1;
    int others = firstNum * (num.size()-1) * pow(10, num.size()-2);
    int next = numOfOne(nextNum);
    return firstDigit + others + next;
}
~~~



### 44.数字序列中某一位的数字

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4

~~~
0         9*10^-1 * 0
1-9       9*10^0 * 1
10-99     9*10^1 * 2
100-999   9*10^2 * 3
exponent位数 则有 9*pow(10,exponent-1)*exponent 个数字
~~~

**解：** 设置循环用n 不断的减去 0, 9, 90, 900, ...

得到exponent即为 目前数字的位数

n变成了 从pow(10, exponent)开始的位数，且**从1开始计数**

例如 1,2,3 所在3位数，但都属于第一个数字 所以(n-1)/exponent为 第几个数字

**从0开始计数**

(n-1)%exponent为该数字的第几位 **从0开始计数**

~~~c++
int num = (n-1)/exponent + pow(10, exponent-1);
int digit = (n-1)%exponent;
~~~

~~~c++
int res = (num/(int)pow(10, exponent-digit-1))%10;
//OR 转换成字符串 更为直观
string num = to_string(num);
int res = num[digit] - '0';
~~~

**具体实现：**

~~~c++
int findNthDigit(int n) {
    int exponent = 0;
    while(n > 9*pow(10, exponent-1)*exponent){
        n = n - 9*pow(10, exponent-1)*exponent;
        exponent++;
    }
    int num = (n-1)/exponent + pow(10, exponent-1);
    int digit = (n-1)%exponent;
    return (num/(int)pow(10, exponent-digit-1))%10;
}
~~~



### 45.把数组排成最小的数

使用sort函数 **记得在compare函数前 加上 static**

判断组合后的 mn<nm ？ true : false； 直接比较合并后的字符串即可

~~~c++
string minNumber(vector<int>& nums) {
    string res;
    if(nums.empty())return "";
    sort(nums.begin(), nums.end(), cmp);
    for(int i=0;i<nums.size();i++){
        res += to_string(nums[i]);
    }
    return res;
}
static bool cmp(int num1, int num2){
    string numof1 = to_string(num1);
    string numof2 = to_string(num2);
    return (numof1+numof2 < numof2+numof1);
}
~~~



### 46.把数字翻译成字符串

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

**解：** 动态规划 或 深度优先遍历

动态规划：

初始dp[0] = 1;(附加的，为了两位数，例如20，dp[2]能记为2) dp[1] = 1;

dp[i] 代表 第i位 有几种翻译方法 因此最后输出的是 dp[num.size()];

从2开始，如果 (num[i-2]=='0' 不允许01出现) 或者 (stoi(num.substr(i-2,2)) > 25)

则说明 i-2 和 i-1 组合的两位数 无效 因此dp[i] = dp[i-1];

否则 有效， 则dp[i] = dp[i-1] + dp[i-2];

 ~~~c++
int translateNum(int num) {
    string sNum = to_string(num);
    int* dp = new int[sNum.size()+1];
    dp[0] = 1;
    dp[1] = 1;
    for(int i=2; i<=sNum.length(); i++){
        if(sNum[i-2]=='0' || stoi(sNum.substr(i-2,2))>25)
            dp[i] = dp[i-1];
        else dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[sNum.length()];
}
 ~~~



### 47.礼物的最大价值

```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

**解：**优化的动态规划，使用一维数组dp 储存每一行

~~~c++
int maxValue(vector<vector<int>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    int* dp = new int[cols];
    memset(dp, 0, cols*sizeof(int));

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(j==0)dp[j] = dp[j] + grid[i][j];
            else dp[j] = max(dp[j-1], dp[j]) + grid[i][j];
        }
    }
    return dp[cols-1];
}
~~~



### 48.最长不含重复字符的子字符串

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

```
输入: "bbbbb"
输出: 1
```

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
```

**解：**采用hash_map 存放已经记录的{char, 下标}

当发现重复是 将目前的开头i 和 重复的char的下标+1  进行比较 选大的

~~~
i = max(i, mapOfChar[s[j]]+1);
~~~

因为char一个字节 0-255，因此用mapOfChar[256]模拟hash_map

~~~c++
int lengthOfLongestSubstring(string s) {
    int mapOfChar[256];
    memset(mapOfChar, -1, sizeof(mapOfChar));
    int maxLength = 0;
    for(int i=0,j=0; j<s.length(); j++){
        if(mapOfChar[s[j]] > -1)
            i = max(i, mapOfChar[s[j]]+1);
        maxLength = max(maxLength, j-i+1);
        mapOfChar[s[j]] = j;
    }
    return maxLength; 
}
~~~



### 49.丑数

我们把只包含因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

**解：**有数组ugly[n]，ugly[0]=1

t2 = 0;

t3 = 0;

t5 = 0; 表示2,3,5分别要乘的位置 ugly[t5];

~~~c++
int nthUglyNumber(int n) {
    int* ugly = new int[n];
    ugly[0] = 1;
    int t2 = 0, t3 = 0, t5 = 0;
    for(int i=1; i<n; i++){
        int m2 = ugly[t2]*2;
        int m3 = ugly[t3]*3;
        int m5 = ugly[t5]*5;
        int minNum = min(m2,min(m3,m5));
        ugly[i] = minNum;
        if(minNum == m2)t2++;
        if(minNum == m3)t3++;
        if(minNum == m5)t5++;
    }

    return ugly[n-1];
}
~~~


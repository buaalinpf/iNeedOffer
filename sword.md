- [剑指Offer](#剑指offer)
    + [2.赋值运算符](#2赋值运算符)
    + [3.数组中重复的数字](#3数组中重复的数字)
    + [4.二维数组中的查找](#4二维数组中的查找)
    + [5.替换空格](#5替换空格)
    + [6.从尾到头打印链表](#6从尾到头打印链表)
    + [7.重建二叉树](#7重建二叉树)
    + [9.用两个栈实现队列](#9用两个栈实现队列)
    + [10.斐波那契数列](#10斐波那契数列)
    + [11.旋转数组的最小数字](#11旋转数组的最小数字)
    + [12.矩阵中的路径](#12矩阵中的路径)
    + [13.机器人的运动范围](#13机器人的运动范围)
    + [14.剪绳子](#14剪绳子)
    + [15.二进制中1的个数](#15二进制中1的个数)
    + [16.数值的整数次方](#16数值的整数次方)
    + [17.打印从1到最大的n位数](#17打印从1到最大的n位数)
    + [18.删除链表的节点](#18删除链表的节点)
    + [19.正则表达式匹配](#19正则表达式匹配)
    + [20.表示数值的字符串](#20表示数值的字符串)
    + [21.调整数组顺序使奇数位于偶数前面](#21调整数组顺序使奇数位于偶数前面)
    + [22.链表中倒数第K个节点](#22链表中倒数第K个节点)
    + [23.链表中环的入口节点](#23链表中环的入口节点)
    + [24.反转链表](#24反转链表)
    + [25.合并两个排序的链表](#25合并两个排序的链表)
    + [26.树的子结构](#26树的子结构)
    + [27.二叉树的镜像](#27二叉树的镜像)
    + [28.对称的二叉树](#28对称的二叉树)
    + [29.顺时针打印矩阵](#29顺时针打印矩阵)
    + [30.包含min(max)函数的栈](#30包含min(max)函数的栈)
    + [31.栈的压入和弹出序列](#31栈的压入和弹出序列)
    + [32.从上打下打印二叉树(二叉树层序遍历)](#32从上打下打印二叉树(二叉树层序遍历))
    + [33.二叉搜索树的后序遍历序列](#33二叉搜索树的后序遍历序列)
    + [34.二叉数中和为某一值的路径](#34二叉数中和为某一值的路径)
    + [35.复杂链表的复制](#35复杂链表的复制)
    + [36.二叉搜索树与双向链表](#36二叉搜索树与双向链表)
    + [37.序列化二叉树](#37序列化二叉树)
    + [38.字符串的排列](#38字符串的排列)
    + [39.数组中出现次数超过一半的数字](#39数组中出现次数超过一半的数字)
    + [40.最小的K个数](#40最小的K个数)
    + [41.数据流中的中位数](#41数据流中的中位数)
    + [42.连续子数组的最大和](#42连续子数组的最大和)
    + [43.1-整数n中1出现的次数](#4311-整数n中1出现的次数)
    + [44.数字序列中某一位的数字](#44数字序列中某一位的数字)
    + [45.把数组排成最小的数](#45把数组排成最小的数)
    + [46.把数字翻译成字符串](#46把数字翻译成字符串)
    + [47.礼物的最大价值](#47礼物的最大价值)
    + [48.最长不含重复字符的子字符串](#48最长不含重复字符的子字符串)
    + [49.丑数](#49丑数)
    + [50.第一个只出现一次的字符](#50第一个只出现一次的字符)
    + [51.数组中的逆序对](#51数组中的逆序对)
    + [52.两个链表的第一个公共节点](#52两个链表的第一个公共节点)
    + [53.1在排序数组中查找数字](#5311在排序数组中查找数字)
    + [53.2 0到n-1中缺失的数字](#5320到n-1中缺失的数字)
    + [54.二叉搜索树的第K大节点](#54二叉搜索树的第K大节点)
    + [55.二叉树的深度](#55二叉树的深度)
    + [55-2.平衡二叉树](#55-2平衡二叉树)
    + [56-1.数组中数字出现的次数](#56-1数组中数字出现的次数)
    + [56-2数组中数字出现的次数2](#56-2数组中数字出现的次数2)
    + [57.和为S的数字](#57和为S的数字)
    + [57-2.和为s的连续正数序列](#57-2和为s的连续正数序列)
    + [58.左翻转字符串](#58左翻转字符串)
    + [59-1.滑动窗口的最大值](#59-1滑动窗口的最大值)
    + [59-2.队列的最大值](#59-2队列的最大值)
    + [60.n个骰子的点数](#60n个骰子的点数)
    + [61.扑克牌中的顺子](#61扑克牌中的顺子)
    + [62.圆圈中最后剩下的数字](#62圆圈中最后剩下的数字)
    + [63.股票的最大利润](#63股票的最大利润)
    + [64.求1+2+...+n](#64求1+2+...+n)
    + [65.不用加减乘除做加法](#65不用加减乘除做加法)
    + [66.构建乘积数组](#66构建乘积数组)
    + [67.字符串转换成整数](#67字符串转换成整数)
    + [68.树的两个节点的最低公共祖先](#68树的两个节点的最低公共祖先)

# 剑指Offer

### 2.赋值运算符

先判断this与str看是不是同一变量的赋值

创建临时对象 与 str相同

然后交换临时对象 和 本对象中的值

~~~c++
CMyString& CMyString::operator =(const CMyString &str){
	if(this != &str){
        CMyString strTemp(str);
        
        char* pTemp = strTemp.m_pData;
        strTemp.m_pData = m_pData;
        m_pData = pTemp;
    }
    return *this;
}
~~~



### 3.数组中重复的数字

将数字i放在对应的i位置上

~~~c++
int findRepeatNumber(vector<int>& nums) {
    if(nums.size()==0)
        return false;
    for(int i=0;i<nums.size();i++)
    {
        while(i!=nums[i])
        {
            if(nums[i]==nums[nums[i]])
                return nums[i];
            else
            {
                swap(nums[i],nums[nums[i]]);
            }
        }
    }
    return false;
}
~~~



### 4.二维数组中的查找

从右上角开始查找

~~~c++
bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
    int length = matrix.size();
    if(length == 0)return false;
    int row = 0;
    int col = matrix[0].size() - 1;
    while(row<length && col>=0){
        if(matrix[row][col]==target)return true;
        if(matrix[row][col] > target)col--;
        else row++;
    }
    return false;
}
~~~



### 5.替换空格

~~~c++
string replaceSpace(string s) {
    int length = s.length();
    int spaceCount = 0;
    for(int i=0;i<length;i++){
        if(s[i]==' ')spaceCount++;
    }
    int last = length+2*spaceCount;
    s.resize(last);
    for(int i=length-1;i>=0;i--){
        if(s[i]!=' ')s[--last] = s[i];
        else {
            s[--last] = '0';
            s[--last] = '2';
            s[--last] = '%';
        }
    }
    return s;
}

void replaceSpace(char *str,int length) {
    if(str==nullptr||length<=0)return;
    int count = 0;
    for(int i=0;i<length;i++){
        if(str[i]==' ')count++;
    }
    char* begin = str + length;
    char* renovate = begin + count*2;
    while(begin>=str){
        if(*begin != ' ')*renovate-- = *begin;
        else{
            *renovate-- = '0';
            *renovate-- = '2';
            *renovate-- = '%';
        }
        begin--;
    }
}
~~~



### 6.从尾到头打印链表

1.用栈

2.或者 如下：先反转链表 再从尾到头输出

~~~c++
vector<int> reversePrint(ListNode* head) {
    vector<int> res;
    if(head==nullptr)return res;
    if(head->next ==nullptr)return {head->val};
    ListNode* first = head;
    ListNode* second = head->next;
    while(second != nullptr){
        ListNode* temp = second->next;
        second->next = first;
        first = second;
        second = temp;
    }
    head->next = nullptr;
    while(first != nullptr){
        res.push_back(first->val);
        first = first->next;
    }
    return res;
}
~~~



### 7.重建二叉树

~~~c++
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    int prestart = 0;
    int preend = preorder.size() - 1;
    int instart = 0;
    int inend = inorder.size() - 1;
    TreeNode* res = buildTree(preorder,inorder,prestart,preend,instart,inend);
    return res;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder, int prestart, int preend, int instart, int inend){
    if(prestart>preend)return nullptr;
    int rootNum = preorder[prestart];
    TreeNode* root = new TreeNode(rootNum);

    int mid = instart;
    while(mid<=inend){
        if(inorder[mid]==rootNum)break;
        mid++;
    }
    int preCount = mid-instart;
    //int inCount = inend - mid;
    root->left = buildTree(preorder,inorder,prestart+1,prestart+preCount,instart,mid-1);
    root->right = buildTree(preorder,inorder,prestart+preCount+1,preend,mid+1,inend);
    return root;
}
~~~



### 9.用两个栈实现队列

一个栈push

另一个栈 如果空 则 先将push栈里的所有 入pop栈，

如果还空，则return -1

~~~c++
class CQueue {
    public:
    CQueue() {

    }

    void appendTail(int value) {
        pushstack.push(value);
    }
    
    int deleteHead() {
        if(popstack.empty()){
            while(!pushstack.empty()){
                popstack.push(pushstack.top());
                pushstack.pop();
            }
        }
        if(popstack.empty())return -1;
        int res = popstack.top();
        popstack.pop();
        return res;
    }
    stack<int> popstack;
    stack<int> pushstack;
};
~~~



### 10.斐波那契数列

~~~c++
int Fibonacci(int n) {
    if(n<2)return n;
    int first = 0;
    int second = 1;
    int res = 0;
    for(int i=2;i<=n;i++){
        res = first + second;
        first = second;
        second = res;
    }
    return res;
}
~~~

**跳台阶：**

~~~c++
int jumpFloor(int number) {
    if(number<0)return -1;
    if(number<3)return number;
    int first = 1;
    int second = 2;
    int res=0;
    for(int i=3;i<=number;i++){
        res = first + second;
        first = second;
        second = res;
    }
    return res;
}
~~~

可以跳1-n台阶

~~~c++
int jumpFloorII(int number) {
    if(number<=0)return 0;
    return pow(2,number-1);
}
~~~



### 11.旋转数组的最小数字

~~~c++
int minArray(vector<int>& numbers) {
    int length = numbers.size();
    if(length==0)return 0;
    int left = 0;
    int right = length-1;
    int mid = 0;
    while(left < right){
        mid = ((right - left)>>1) + left;
        if(numbers[mid]>numbers[right])left = mid + 1;
        else if(numbers[mid]==numbers[right])right = right - 1;
        else right = mid;
    }
    return numbers[left];
}
~~~



### 12.矩阵中的路径

~~~c++
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        if(board.empty() || word.empty())return false;
        int row = board.size();
        int col = board[0].size();
        bool** visited = new bool*[row];
        for(int i=0;i<row;i++){
            visited[i] = new bool[col];
        }
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                visited[i][j] = false;
            }
        }
        bool res = false;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                res = coreExist(board,word,0,i,j,row,col,visited);
                if(res)return res;
            }
        }
        return res;
    }
    bool coreExist(vector<vector<char>>& board, string word, int index, int row, int col, int rows, int cols, bool** visited){
        if(index >= word.length())return true;
        if(row<0 || row>=rows || col<0 || col>=cols)return false;
        bool res = false;
        if((!visited[row][col]) && board[row][col] == word[index]){
            visited[row][col] = true;
            res = coreExist(board,word,index+1,row-1,col,rows,cols,visited)
            || coreExist(board,word,index+1,row+1,col,rows,cols,visited)
            || coreExist(board,word,index+1,row,col-1,rows,cols,visited)
            || coreExist(board,word,index+1,row,col+1,rows,cols,visited);
            if(!res)visited[row][col] = false;
        }
        return res;
    }
};
~~~



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



### 14.剪绳子

贪心算法：

~~~c++
int cutRope(int number) {
    if(number<=1)return 0;
    if(number==2)return 1;
    if(number==3)return 2;
    int timesOf3 = number/3;
    int remainder = number%3;

    if(remainder==1){
        timesOf3--;
    }
    int timesOf2 = (number-timesOf3*3)/2;
    return pow(3,timesOf3)*pow(2,timesOf2);
}
~~~

动态规划：

~~~c++
int cutRope(int number) {
    if(number<=1)return 0;
    if(number==2)return 1;
    if(number==3)return 2;
    int* products = new int[number+1];
    products[0]=0;
    products[1]=1;
    products[2]=2;
    products[3]=3;
    int maxi=0;
    for(int i=4;i<=number;i++){
        for(int j=1;j<=i/2;j++){
            int product = products[j]*products[i-j];
            if(product>maxi){
                maxi = product;
            }
        }
        products[i]=maxi;
    }
    int max = products[number];
    delete[] products;
    return max;
}
~~~



### 15.二进制中1的个数

~~~c++
int hammingWeight(uint32_t n) {
    int res = 0;
    while(n){
        res++;
        n = n & (n-1);
    }
    return res;
}
~~~



### 16.数值的整数次方

~~~c++
double myPow(double x, int n) {
    if(x==0)return 0;
    if(n==0)return 1;
    int flag = 0;
    if(n>0)flag = 1;
    else n = -(n+1);
    double res = myPowCore(x, n);
    if(flag==1)return res;
    else return 1/(res*x);
}
double myPowCore(double x, int n){
    if(n==0)return 1;
    if(n==1)return x;
    double midPow = myPowCore(x,n>>1);
    if(n&1)return midPow*midPow*x;
    else return midPow*midPow;
}
~~~



### 17.打印从1到最大的n位数

全排列n位数

将前面的0去除 譬如0001 0002 9999

输出 1 2  9999

### 18.删除链表的节点

~~~c++
ListNode* deleteNode(ListNode* head, int val) {
    if(head==nullptr)return nullptr;
    ListNode* res = new ListNode(0);
    res->next = head;
    ListNode* pHead = res; 
    while(pHead->next){
        if(pHead->next->val == val){
            pHead->next = pHead->next->next;
            break;
        }
        pHead = pHead->next;
    }
    return res->next;
}
~~~



### 19.正则表达式匹配

~~~c++
bool isMatch(string s, string p) {
    if (p.empty()) return s.empty();
    if (p.size() > 1 && p[1] == '*') {
        return isMatch(s, p.substr(2)) || (!s.empty() && (s[0] == p[0] || p[0] == '.') && isMatch(s.substr(1), p));
    }
    return !s.empty() && (s[0] == p[0] || p[0] == '.') && isMatch(s.substr(1), p.substr(1));
}
~~~



### 20.表示数值的字符串

~~~c++
bool isNumber(string s) {
    if(s.empty())return false;
    int start = 0;
    int end = s.length();
    while(start<end && s[start]==' ')start++;
    if(start<end && (s[start]=='+' || s[start]=='-'))start++;
    bool numeric = isInt(s, start);
    if(start<end && s[start]=='.'){
        start++;
        numeric = isInt(s, start) || numeric;
    }
    if(start<end && (s[start]=='E' || s[start]=='e')){
        start++;
        if(start<end && (s[start]=='+' || s[start]=='-'))start++;
        numeric = numeric && isInt(s, start);
    }
    while(start<end && s[start]==' ')start++;
    return numeric && (start==end);
}
bool isInt(string s, int &start){
    int temp = start;
    for(int i=start;i<s.length();i++){
        if(s[i]>='0' && s[i]<='9')start++;
        else break;
    }
    return start>temp;
}
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
vector<int> exchange(vector<int>& nums) {
    int start = -1;
    for(int i=0;i<nums.size();i++){
        if(nums[i]&1 == 1){
            start++;
            if(i>start){
                swap(nums[i],nums[start]);
            }
        }
    }
    return nums;
}
~~~



### 22.链表中倒数第K个节点

双指针，1->2->3->4->5 若求倒数第二个 即一个指针指向1 一个指针指向2

~~~c++
ListNode* getKthFromEnd(ListNode* head, int k) {
    if(head == nullptr)return nullptr;
    ListNode* pRes = head;
    ListNode* pNode = head;
    while(k>1 && pNode){
        pNode = pNode->next;
        k--;
    }
    if(pNode == nullptr)return nullptr;
    while(pNode->next){
        pNode = pNode->next;
        pRes = pRes->next;
    }
    return pRes;
}
~~~

注意：记得判断k>len 则返回nullptr



### 23.链表中环的入口节点

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null

1.双指针，一个指针走1步 一个指针走2步， 得到重合位置

2.从重合位置 循环 再次到达重合位置，计算环中节点数count

3.meet节点从head节点向后count次next，然后meet==head时 即为入口节点

~~~c++
ListNode* EntryNodeOfLoop(ListNode* pHead)
{
    ListNode* meet = meetingNode(pHead);
    if(meet == nullptr)return nullptr;
    int count = 1;
    ListNode* copy = meet->next;
    while(copy!=meet){
        count++;
        copy = copy->next;
    }
    meet = pHead;
    while(count>0){
        meet = meet->next;
        count--;
    }
    while(meet !=pHead){
        meet = meet->next;
        pHead = pHead->next;
    }
    return meet;
}
ListNode* meetingNode(ListNode* pHead){
    ListNode* pNode1 = pHead;
    ListNode* pNode2 = pHead;
    while(pNode2){
        pNode2 = pNode2->next;
        if(pNode2 == nullptr)break;
        if(pNode1 == pNode2)return pNode1;
        pNode1 = pNode1->next;
        pNode2 = pNode2->next;
    }
    return nullptr;
}
~~~



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

**直接在原树上修改节点：不用创建新的节点，时间更快**

~~~c++
TreeNode* invertTree(TreeNode* root) {
    invertCore(root);
    return root;
}
void invertCore(TreeNode* root){
    if(root == nullptr)return;
    TreeNode* temp = root->left;
    root->left = root->right;
    root->right = temp;
    invertCore(root->left);
    invertCore(root->right);
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



### 32.从上打下打印二叉树(二叉树层序遍历)

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
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        if(root == nullptr)return{};
        dfs(root, sum);
        return res;
    }
    void dfs(TreeNode* root, int sum){
        temp.push_back(root->val);
        if(!root->left && !root->right && root->val == sum){
            res.push_back(temp);
        }
        if(root->left)dfs(root->left, sum-root->val);
        if(root->right)dfs(root->right, sum-root->val);
        
        temp.pop_back();
    }
private:
    vector<vector<int>> res;
    vector<int> temp;
};
~~~



### 35.复杂链表的复制

使用unordered_map<oldNode\*, newNode\*>记录新建的Node和原Node的关系

先遍历一遍链表，得到完整的map

再遍历一遍:

~~~c++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if(head == nullptr)return nullptr;
        unordered_map<Node*, Node*> mapOfNodes;
        Node* pNode = head;
        Node* newNode = new Node(head->val);
        mapOfNodes.insert({head, newNode});
        Node* res = newNode;
        pNode = pNode->next;
        while(pNode){
            newNode = new Node(pNode->val);
            mapOfNodes.insert({pNode, newNode});
            pNode = pNode->next;
        }
        pNode = head;
        while(pNode){
            newNode = mapOfNodes[pNode];
            if(pNode->next)newNode->next = mapOfNodes[pNode->next];
            if(pNode->random)newNode->random = mapOfNodes[pNode->random];
            pNode = pNode->next;
        }
        return res;
    }
};
~~~



### 36.二叉搜索树与双向链表

**解：**设置Node* lastNode; 用于表示已经得到的链表的最后一个节点

中序遍历，左根右，当遍历到根节点时，使得root和lastNode互相指向对方

更新lastNode = root

一开始设置lastNode = nullptr， 保证链表头节点指向nullptr

主函数：遍历出最左节点得到head,然后让head和得到的lastNode相互指向

~~~c++
class Solution {
public:
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
};
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
class Solution {
public:
    vector<string> permutation(string s) {
        sort(s.begin(),s.end());
        corePermutation(s, 0);
        return res;
    }
    void corePermutation(string s, int index){
        if(index == s.length()){
            res.push_back(s);
            return;
        }
        for(int i=index;i<s.length();i++){
            if(i!=index && s[i] == s[index])continue;
            swap(s[i], s[index]);
            corePermutation(s, index+1);
            //swap(s[i], s[index]);
        }
    }
private:
    vector<string> res;
};
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

**解：**利用patition函数 直到mid = k-1为止

~~~c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        if(arr.empty() || k>arr.size() || k<=0)return {};
        vector<int> res;
        int start = 0;
        int end = arr.size() - 1;
        int mid = partition(arr, start, end);
        while(mid != k-1){
            if(mid<k-1)
                start = mid + 1;
            else end = mid - 1;
            mid = partition(arr, start, end);
        }
        for(int i=0;i<k;i++){
            res.push_back(arr[i]);
        }
        return res;
    }
    int randomNum(int start, int end){
        std::srand(std::time(nullptr));
        return start + std::rand()/((RAND_MAX + 1u)/(end - start + 1));
    }
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
};
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
class MedianFinder {
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    
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
private:
    vector<int> maxHeap;
    vector<int> minHeap;
};
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

临届判断:num.size()==1 :
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

### 50.第一个只出现一次的字符

一次遍历建立hash_map 存放字符和出现的次数

再次遍历 当第一次得到次数为1是 即为第一个出现的字符

~~~c++
int FirstNotRepeatingChar(string str) {
    if(str.empty())return -1;
    unordered_map<char, int> timesOfChar;
    for(int i=0;i<str.length();i++){
        if(timesOfChar.find(str[i])==timesOfChar.end())
            timesOfChar.insert({str[i], 1});
        else timesOfChar[str[i]]++;
    }
    for(int i=0;i<str.length();i++){
        if(timesOfChar[str[i]]==1)return i;
    }
    return -1;
}
~~~



### 51.数组中的逆序对

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

~~~
输入: [7,5,6,4]
输出: 5
~~~

**解：**归并排序

~~~c++
int reversePairs(vector<int>& nums) {
    if(nums.empty())return 0;
    vector<int> copy(nums.size());
    int res = 0;
    merge(nums, copy, 0, nums.size()-1, res);
    return res;
}
void merge(vector<int>& nums, vector<int>& copy, int start, int end, int &res){
    if(start >= end)return;
    int mid = start + (end - start)/2;
    merge(nums,copy,start,mid,res);
    merge(nums,copy,mid+1,end,res);
    int i=start, j=mid+1, k=start;
    while(i <= mid && j <= end){
        if(nums[j] < nums[i]){
            copy[k++] = nums[j++];
            res += mid-i+1;//与归并排序唯一添加的一行代码
        }else {
            copy[k++] = nums[i++];
        }
    }
    while(i<=mid)copy[k++] = nums[i++];
    while(j<=end)copy[k++] = nums[j++];
    for(int index=start; index<=end; index++){
        nums[index] = copy[index];
    }
}
~~~



### 52.两个链表的第一个公共节点

计算两个链表的长度

根据长度差，把较长的，先next (len1-len2)次

然后同时向后，即能遍历到重合点

~~~c++
ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
    if(pHead1==nullptr || pHead2==nullptr)return nullptr;
    ListNode* pCopy1 = pHead1;
    ListNode* pCopy2 = pHead2;
    int len1 = getLength(pCopy1);
    int len2 = getLength(pCopy2);
    if(len1 > len2){
        while(len1 - len2){
            pHead1 = pHead1->next;
            len1--;
        }
    }else{
        while(len2 - len1){
            pHead2 = pHead2->next;
            len2--;
        }
    }
    while(pHead1 && pHead2){
        if(pHead1 == pHead2)return pHead1;
        pHead1 = pHead1 -> next;
        pHead2 = pHead2 -> next;
    }
    return nullptr;
}
int getLength(ListNode* pHead){
    int len = 0;
    while(pHead){
        len++;
        pHead = pHead -> next;
    }
    return len;
}
~~~

**双指针法：**

当一个链表的指针指向末尾时 转向**另一个链表的头结点**

相当于 遍历 A+B次  和  B+A次 

如果长度相同：遍历第一次即能找到重合点

如果长度不一样：一定会在第二次遍历重合

如果无重合：**第二次遍历后，两指针皆为nullptr，则循环结束**

~~~c++
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    if(!headA || !headB)return nullptr; 
    ListNode* indexA = headA;
    ListNode* indexB = headB;
    while(indexA != indexB){
        indexA = indexA==nullptr?headB:indexA->next;
        indexB = indexB==nullptr?headA:indexB->next;
    }
    return indexA;
}
~~~



### 53.1在排序数组中查找数字

~~~c++
int GetNumberOfK(vector<int> data ,int k) {
    if(data.empty())return 0;
    int start = 0;
    int end = data.size() - 1;
    int left = getFirstK(data,k,start,end);
    int right = getLastK(data,k,start,end);
    if(left!=-1 && right!=-1)return right-left+1;
    else return 0;
}
int getFirstK(vector<int> data, int k, int start, int end){
    if(start>end)return -1;
    int mid = (end + start)/2;
    if(data[mid]<k)start = mid + 1;
    if(data[mid]>k)end = mid - 1;
    if(data[mid]==k)
        if(mid==0 || (mid>0 && data[mid-1]<k))return mid;
    	else end = mid - 1;
    return getFirstK(data,k,start,end);
}
int getLastK(vector<int> data, int k, int start, int end){
    if(start>end)return -1;
    int mid = (end + start)/2;
    if(data[mid]<k)start = mid + 1;
    if(data[mid]>k)end = mid - 1;
    if(data[mid]==k)
        if(mid==data.size()-1 || (mid<data.size()-1 && data[mid+1]>k))return mid;
    	else start = mid + 1;
    return getLastK(data,k,start,end);
}
~~~

### 53.2 0到n-1中缺失的数字

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

~~~
输入: [0,1,3]
输出: 2
~~~

二分查找，检查nums[mid] == mid?

如果等于 那缺失的数字一定位于右侧

如果不等于 需要判断前一个数字nums[mid-1] == mid - 1? 或者 mid==0

如果前一个数字 不满足 那正好是缺失的位置mid 为 缺失的数字

否则 继续向前查找

~~~c++
int missingNumber(vector<int>& nums) {
    int left = 0;
    int right = nums.size() - 1;
    while(left <= right){
        int mid = left + (right - left)/2;
        if(nums[mid] != mid){
            if(mid==0 || nums[mid-1] == mid-1){
                return mid;
            }else right = mid - 1;
        }else left = mid + 1;
    }
    return nums.size();
}
~~~



### 54.二叉搜索树的第K大节点

~~~c++
TreeNode* KthNode(TreeNode* pRoot, int k)
{
    if(pRoot == nullptr || k<1)return nullptr;
    TreeNode* res = nullptr;
    inOrder(pRoot,k,res);
    return res;
}
void inOrder(TreeNode* pRoot, int &k, TreeNode* &res){
    if(pRoot == nullptr || res != nullptr)return;
    inOrder(pRoot->left, k, res);
    if(k==1){
        res = pRoot;
    }
    k--;
    inOrder(pRoot->right, k, res);
}
~~~



### 55.二叉树的深度

~~~c++
int TreeDepth(TreeNode* pRoot)
{
    if(pRoot == nullptr)return 0;
    int left = TreeDepth(pRoot->left);
    int right = TreeDepth(pRoot->right);
    int res = (left > right) ? left : right;
    return res + 1;
}
~~~

### 55-2.平衡二叉树

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



### 56-1.数组中数字出现的次数

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。

~~~
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
~~~

**解：**通过异或的方法，异或所有数字，则得到的结果是 **两个答案的异或**

因为**相同的数字 异或为0 互相抵消**

利用getDigit函数 得到两个答案异或的 **第一个为1的位**，说明这两个数字在这个位上是不同的

因此将数组中所有的数字 按照 **该位是否为1** 可以**划分成两波**，**互相异或** 则得到两个数字

~~~c++
vector<int> singleNumbers(vector<int>& nums) {
    int num = 0;
    for(int i=0; i<nums.size(); i++){
        num = num ^ nums[i];
    }
    int num1 = 0;
    int num2 = 0;
    int digitNum = getDigit(num);
    if(digitNum == -1)return {};
    for(int i=0; i<nums.size(); i++){
        if((nums[i] & digitNum) == 0)num1 ^= nums[i];
        else num2 ^= nums[i];
    }
    return {num1, num2};
}
int getDigit(int num){
    int index = 0;
    int res = 0;
    while(index < 8*sizeof(int)){
        res = 1 << index;
        if((num & res) != 0)return res;
        index++;
    }
    return -1;
}
~~~

### 56-2数组中数字出现的次数2

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

~~~
输入：nums = [3,4,3,3]
输出：4
~~~

**解：**将每个数字的每一位相加，然后对每一位求 %3，即能够得到答案的每一位

然后再通过移位计算答案:

~~~c++
for(int i=31; i>=0; i--){
    res = res << 1;
    res += digitNum[i]%3;
}
~~~



~~~c++
int singleNumber(vector<int>& nums) {
    int digitNum[32];
    memset(digitNum, 0, sizeof(digitNum));
    for(int i=0; i<nums.size(); i++){
        addNum(digitNum, nums[i]);
    }
    int res = 0;
    for(int i=31; i>=0; i--){
        res = res << 1;
        res += digitNum[i]%3;
    }
    return res;
}
void addNum(int* digitNum, int num){
    int digit = 1;
    int index = 0;
    while(index < 32){
        digit = 1 << index;
        digitNum[index] += (num & digit)==0 ? 0 : 1;
        index++;
    }
}
~~~

~~~c++
int singleNumber(vector<int>& nums) {
    int ans = 0;
    for (int i = 0; i < 32; i++){
        int count = 0;
        for (auto n : nums){
            if ((1 << i ) & n) count++;
        }
        if (count % 3) ans += (1 << i);
    }
    return ans;
}
~~~



### 57.和为S的数字

~~~c++
vector<int> FindNumbersWithSum(vector<int> array,int sum) {
    vector<int> res;
    if(array.size() < 2)return res;
    int left = 0;
    int right = array.size()-1;
    while(left < right){
        if(array[left]+array[right]==sum){
            res.push_back(array[left]);
            res.push_back(array[right]);
            break;
        }
        if(array[left]+array[right]<sum){
            left++;
        }else{
            right--;
        }
    }
    return res;
}
~~~



### 57-2.和为s的连续正数序列

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）

~~~
输入：target = 9
输出：[[2,3,4],[4,5]]

输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
~~~

**解：**定义small = 1; big = 2;

[small, big]的和 如果大于target 则 small++;

小于 则 big++；

等于 small++; big++;

~~~
模拟target = 9
1 2
1 2 3 4 > 9 small++;
2 3 4 == 9 small++; big++;
3 4 5 > 9 small++;
4 5 == 9 small++; big++; small == 5;结束
~~~



~~~c++
vector<vector<int>> findContinuousSequence(int target) {
    int small = 1;
    int big = 2;
    int cur = small + big;
    int limit = (target + 1)>>1;
    vector<int> temp;
    vector<vector<int>> res;
    while(small < limit){
        if(cur == target){
            for(int i=small; i<=big; i++)
                temp.push_back(i);
            res.push_back(temp);
            temp.clear();
            cur -= small;
            small++;
            big++;
            cur += big;
        }else if(cur > target){
            cur -= small;
            small++;
        }else {
            big++;
            cur += big;
        }
    }
    return res;
}
~~~



### 58.左翻转字符串

**”abcXYZdef” -  “XYZdefabc”**

1.交换

~~~c++
string LeftRotateString(string str, int n) {
    if(str.empty() || n < 1 || n>=str.length())return str;
    auto begin = str.begin();
    auto end = next(begin,n);
    while(begin != end){
        swap(*begin, *end);
        begin++;
        if(end != str.end()-1)end++;
    }
    return str;
}
~~~

2.右加左

~~~c++
string reverseLeftWords(string s, int n) {
    return s.substr(n) + s.substr(0, n);
}
~~~

3.翻转，再从右往前数n处 各自翻转两侧

~~~c++
string reverseLeftWords(string s, int n) {
    if(s.empty())return s;
    reverse(s.begin(), s.end());
    auto mid = prev(s.end(), n);
    reverse(s.begin(), mid);
    reverse(mid, s.end());
    return s;
}
~~~



**翻转字符串“student. a am I”**

~~~c++
string ReverseSentence(string str) {
    if(str.empty())return str;
    reverse(str.begin(),str.end());
    auto begin = str.begin();
    auto end = str.begin();
    while(begin!=str.end()){
        if(*begin == ' '){
            begin++;
            end++;
        }else if(*end == ' ' || end == str.end()){
            reverse(begin,end);
            begin = end;
        }else end++;
    }
    return str;
}
~~~

考虑多余空格：要删去最左和最右的空格，以及单词之间1个以上的空格：

~~~c++
string reverseWords(string s) {
    if(s.empty())return s;
    string res;
    reverse(s.begin(), s.end());
    auto start = s.begin();
    auto end = start;
    while(start < s.end()){
        while(start < s.end() && (*start) == ' ')
            start++;
        end = start;
        while(end < s.end() && (*end) != ' ')
            end++;
        reverse(start, end);
        if(start != end)res += s.substr(start-s.begin(), end - start) + ' ';
        start = next(end);
    }
    return res.substr(0, res.length()-1);
}
~~~



### 59-1.滑动窗口的最大值

~~~
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
~~~

**解：**建立队列 存放 数组下标，为了判读**是否在窗口外**

当新数字 > nums[deque.back()] 则需要一直删除 直到 没有比他大的为止

<= 都需要 push_back， 因为当前面的弹出窗口，依旧有可能成为最大值



~~~c++
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    vector<int> res;
    deque<int> dequeOfNums;
    for(int i=0; i<nums.size(); i++){
        while(!dequeOfNums.empty() && i - dequeOfNums.front() >= k){
            dequeOfNums.pop_front();
        }
        while(!dequeOfNums.empty() && nums[i] > nums[dequeOfNums.back()]){
            dequeOfNums.pop_back();
        }
        dequeOfNums.push_back(i);
        if(i >= k-1)res.push_back(nums[dequeOfNums.front()]);
    }
    return res;
}
~~~



### 59-2.队列的最大值

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

**解：**需要两个队列，一个是正常队列 normal，另一个是存储最大值 maxDeque

与滑动窗口类似，只有value > maxDeque.back()时 需要将之前 小于value的都删除

当pop_front()的时候，只需要比较 此时的值 和 maxDeque中的是否相等

相等则需要 删去 maxDeque的值

~~~c++
class MaxQueue {
public:
    MaxQueue() {

    }
    
    int max_value() {
        if(maxDeque.empty())return -1;
        else return maxDeque.front();
    }
    
    void push_back(int value) {
        normal.push_back(value);
        while(!maxDeque.empty() && maxDeque.back() < value)
            maxDeque.pop_back();
        maxDeque.push_back(value);
    }
    
    int pop_front() {
        if(normal.empty())return -1;
        if(normal.front() == maxDeque.front())
            maxDeque.pop_front();
        int res = normal.front();
        normal.pop_front();
        return res;
    }
private:
    deque<int> normal;
    deque<int> maxDeque;
};
~~~



### 60.n个骰子的点数

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

~~~
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
~~~

**解：**使用两个数组 交替存储 动态规划

~~~
1 - 6
2 - 12
3 - 18
4 - 24
~~~

1-cur中每个元素是 cur 中的 n-6, n-5, n-4, n-3, n-2, n-1 相加的结果

~~~c++
vector<double> twoSum(int n) {
    vector<int> sup[2];
    vector<double> res;
    int total = pow(6, n);
    sup[0] = vector<int>(6*n+1, 0);
    sup[1] = vector<int>(6*n+1, 0);
    int cur = 0;
    for(int i=1; i<=6; i++){
        sup[cur][i] = 1;
    }
    for(int i=2; i<=n; i++){
        for(int j=1; j<i-1; j++){//清除掉上一层(n-1)之前无效的数字
            sup[cur][j] = 0;
        }
        for(int j=i; j<=6*i; j++){
            sup[1-cur][j] = 0;
            for(int k=j-6; k<=j-1; k++){
                if(k>=1)sup[1-cur][j] += sup[cur][k]; 
            }
        }
        cur = 1 - cur;
    }
    for(int i=n; i<=6*n; i++){
        res.push_back((double)sup[cur][i]/total);
    }
    return res;
}
~~~



### 61.扑克牌中的顺子

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

~~~
输入: [0,0,1,2,5]
输出: True
~~~

**解：**先排序数组

然后得到前面0的个数，即大小王的个数

然后遍历后面的：

如果 nums[i+1] == nums[i]，直接false

然后 gaps += nums[i+1] - nums[i] - 1; 如果连续，则加0；不连续 加上缺少的数字个数

~~~c++
bool IsContinuous( vector<int> numbers ) {
    int length = numbers.size();
    if(length==0)return false;
    sort(numbers.begin(),numbers.end());
    int zeronum = 0;
    while(numbers[zeronum]==0){
        zeronum++;
    }
    int gaps = 0;
    for(int i=zeronum+1;i<length;i++){
        if(numbers[i] == numbers[i-1])return false;
        gaps += numbers[i] - numbers[i-1] - 1;
    }
    return (gaps<=zeronum)?true:false;
}
~~~



### 62.圆圈中最后剩下的数字

0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。



~~~c++
int LastRemaining_Solution(int n, int m)
{
    if(n<1)return -1;
    if(n==1)return 0;
    return (LastRemaining_Solution(n-1,m)+m)%n;
}
~~~

~~~c++
int lastRemaining(int n, int m) {
    if(m < 1 || n < 1)return -1;
    int* dp = new int[n+1];
    dp[1] = 0;
    for(int i=2; i<=n; i++){
        dp[i] = (dp[i-1] + m)%i;
    }
    return dp[n];
}
~~~

~~~c++
int LastRemaining_Solution(int n, int m)
{
    if(n<1)return -1;
    list<int> circle;
    for(int i=0;i<n;i++){
        circle.push_back(i);
    }
    auto current = circle.begin();
    while(circle.size()>1){
        for(int i=1;i<m;i++){
            current++;
            if(current==circle.end())current = circle.begin();
        }
        current = circle.erase(current);
        if(current == circle.end())current = circle.begin();
    }
    return *current;
}
~~~



### 63.股票的最大利润

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

~~~
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
~~~

**解：**

记录cur = 0，动态规划思路

cur = cur + prices[i] - prices[i-1] 

如果大于0，则当前最大利润为cur

小于等于0，则当前最大利润为0

~~~c++
int maxProfit(vector<int>& prices) {
    int cur = 0;
    int maxprofit = 0;
    for(int i=1; i<prices.size(); i++){
        cur += prices[i] - prices[i-1];
        if(cur < 0)cur = 0;
        if(maxprofit < cur)maxprofit = cur;
    }
    return maxprofit;
}
~~~

记录最小的支出put

每次拿盈利get 和 **最小支出加当前卖出**比较 get = max(get, **put+prices[i]**) 

~~~c++
int maxProfit(vector<int>& prices) {
    if(prices.size() < 2)return 0;
    int put = -prices[0];
    int get = 0;
    for(int i=1; i<prices.size(); i++){
        get = max(get, put + prices[i]);
        put = max(put, -prices[i]);
    }
    return get;
}
~~~



### 64.求1+2+...+n

利用短路特性：

当n == 0 时 自动返回0 不会继续实行 &&后的部分

~~~c++
int sumNums(int n) {
    n && (n += sumNums(n-1));
    return n;
}
~~~



~~~c++
class assist{
public:
    assist() {N++;sum += N;}
    static void reset(){N=0;sum=0;}
    static unsigned int GetSum(){return sum;}
private:
    static int N;
    static int sum;
};
int assist::N = 0;
int assist::sum = 0;
class Solution {
public:
    int Sum_Solution(int n) {
        assist::reset();
        assist * p = new assist[n];
        delete []p;
        p = nullptr;
        return assist::GetSum();
    }
};
~~~



### 65.不用加减乘除做加法

~~~c++
int Add(int num1, int num2)
{
    while(num2!=0){
        int xornum = num1 ^ num2;
        int andnum = num1 & num2;
        andnum = andnum << 1;
        num1 = xornum;
        num2 = andnum;
    }
    return num1;
}
~~~

存在负数情况：c++不允许int负数左移，因此要转换成 **unsigned int**

~~~c++
int add(int a, int b) {
    while(b){
        int carry = (unsigned int)(a & b) << 1;
        a = a ^ b;
        b = carry;
    }
    return a;
}
~~~



### 66.构建乘积数组

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

**解：**在纸上画个n-1*n-1的矩阵 将B[i] 分成 C[0, i-1] 和 D[i+1, n-1]两部分

~~~c++
vector<int> multiply(const vector<int>& A) {
    int n = A.size();
    vector<int> res(n,1);
    for(int i=1;i<n;i++){
        res[i] = res[i-1] * A[i-1];
    }
    int temp = 1;
    for(int i=n-2;i>=0;i--){
        temp = temp * A[i+1];
        res[i] = res[i] * temp;
    }
    return res;
}
~~~

更快：

~~~c++
vector<int> constructArr(vector<int>& a) {
    vector<int> re(a.size());
    int sum=1;
    for(int i = 0;i<a.size();i++)
    {
        re[i]=sum;
        sum*=a[i];
    }
    sum=1;
    for(int i = a.size()-1;i>=0;i--)
    {
        re[i]*=sum;
        sum*=a[i];
    }
    return re;
}
~~~



### 67.字符串转换成整数

~~~c++
int StrToInt(string str) {
    if(str.empty())return 0;
    int flag = 1;
    int start = 0;
    long res = 0;
    while(str[start]==' ' && start<str.size()){
        start++;
    }
    if(start == str.size())return 0;
    if(str[start] == '-'){flag = -1;start++;}
    else if(str[start] == '+')start++;
    for(int i = start;i<str.size();i++){
        if(str[i]>'9' || str[i]<'0')return 0;
        res = res*10 + str[i]-'0';
        if(flag==1 && res>2147483647){res = 0;break;}
        else if(flag==-1 && res>2147483648){res = 0;break;}
    }
    return (int)(flag*res);
}
~~~

不使用long的情况：

~~~c++
int strToInt(string str) {
    int length = str.length();
    if(length ==0)return 0;
    int index = 0;
    int flag = 0;
    int res = 0;
    while(index < length && str[index] == ' ')index++;
    if(str[index] == '-'){
        flag = 1;
        index++;
    }else if(str[index] == '+')index++;
    while(index < length && str[index] >= '0' && str[index] <= '9'){
        if(res > INT_MIN/10)res = res*10 - (str[index] - '0');
        else if(res == INT_MIN/10){
            if(str[index] - '0' < 9)res = res*10 - (str[index] - '0');
            else return flag==0 ? INT_MAX : INT_MIN;
        }else return flag==0 ? INT_MAX : INT_MIN;
        index++;
    }
    if(flag == 1)return res;
    if(flag == 0 && res == INT_MIN)return INT_MAX;
    return -res;
}
~~~



### 68.树的两个节点的最低公共祖先

二叉搜索树：

~~~c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if(root == nullptr)return nullptr;

    if(root->val < p->val && root->val < q->val)
        return lowestCommonAncestor(root->right, p, q);
    if(root->val > p->val && root->val > q->val)
        return lowestCommonAncestor(root->left, p, q);
    return root;
}
~~~

普通二叉树：

后序遍历：

~~~c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if(!root || root == p || root == q) return root;
    auto left = lowestCommonAncestor(root->left, p, q);
    auto right = lowestCommonAncestor(root->right, p, q);
    if(!left) return right;
    if(!right) return left;
    return root;      
}
~~~




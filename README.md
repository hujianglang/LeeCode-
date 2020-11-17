# LeeCode- Java
记录Leecode刷题思路与知识点

## 题目分类

# Hash相关

- [(1)两数之和](/src/hash相关/q1_两数之和)

**题目描述：**

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例:**

```html
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

**Solution:**

```java
public class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                return new int[]{map.get(nums[i]), i};
            }
            map.put(target - nums[i], i);
        }
        return null;
    }
}
```



- [(2)字符串中的第一个唯一字符](/src/hash相关/q387_字符串中的第一个唯一字符)

```java
//字符串中的第一个唯一字符
//给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。
//示例：
//s = “leetcode” 返回 0 ;
// s = “loveleetcode”  返回 2
//提示：你可以假定该字符串只包含小写字母。

/*
解题思路：#
很简单的题，无非就是对字符串的字母进行频率统计，找到出现频率为1 的字母索引。
借助哈希映射两次遍历完成。第一次遍历进行字母频率统计，Hash Map 的Key 为字母，
Value 为出现频率。第二次遍历找到频率为 1 的字母索引返回即可。
不同于单词频率统计，字母一共只有 26 个，所以可以直接利用 ASii 码表里小写字母数值从 97～122，
直接用 int 型数组映射。建立映射：索引为 小写字母的 ASii 码值，存储值为出现频率。
 */
class Solution{
    public int firstUniqChar(String s){
        char[] chars = s.toCharArray();//转成char数组
        Map<Character,Integer> map = new HashMap<>();
        for(Character c: chars) map.put(c,map.getOrDefault(c,0) + 1);//频率统计
        for(int i = 0; i < chars.length; i++){
            if(map.get(chars[i]) == 1) return i;
        }
        return -1;
    }
}

//Solution2 : 数组映射解题
class Solution2{
    public int firstUniqChar(String s){
        char[] chars = s.toCharArray();
        int base = 97;
        int[] loc = new int[26];
        for(char c: chars) loc[c-base] += 1;
        for(int i = 0; i < chars.length; i++)
        {
            if(loc[chars[i] - base] == 1)
                return i;
        }
        return -1;
    }
}

//Solution3:利用字符串集成操作函数解题，效率高巧妙
/*
indexOf():返回元素第一次出现的索引，没有则返回-1
lastIndexOf():返回元素最后一次出现的索引，没有则返回-1
 */
class Solution3{
    public int firstUniqChar(String s){
        int res = s.length();
        for(int i = 'a'; i <= 'z'; i++){
            int firstIndex = s.indexOf((char)i);
            if(firstIndex == -1) continue;
            int lastIndex = s.lastIndexOf((char)i);
            if(firstIndex == lastIndex){
                res = Math.min(firstIndex,res);
            }
        }
        return res == s.length() ? -1 : res;
    }
}
```


# 链表操作

- [q2_两数相加](/src/链表操作/q2_两数相加)

### 1.两数相加

**题目描述：**

给出两个 **非空** 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 **逆序** 的方式存储的，并且它们的每个节点只能存储 **一位** 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例：**

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

**节点结构：**

```java
public class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}
```

**Solution:**

```java
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy;
        while (l1 != null || l2 != null || carry == 1) {
            pre.next = new ListNode(carry);
            pre = pre.next;
            if (l1 != null) {
                pre.val += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                pre.val += l2.val;
                l2 = l2.next;
            }
            carry = pre.val / 10;
            pre.val %= 10;
        }
        return dummy.next;
    }
}
```



- [q19_删除链表的倒数第N个节点](/src/链表操作/q19_删除链表的倒数第N个节点)
### 2.删除链表的倒数第N个节点

**题目描述：**

给定一个链表，删除链表的倒数第 *n* 个节点，并且返回链表的头结点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**说明：**

给定的 *n* 保证是有效的。

**进阶：**

你能尝试使用一趟扫描实现吗？

**Solution：**

```java
public class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = head;
        ListNode slow = dummy;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
```


- [q25_k个一组翻转链表](/src/链表操作/q25_k个一组翻转链表)


- [q61_旋转链表](/src/链表操作/q61_旋转链表)

- [q138_复制带随机指针的链表](/src/链表操作/q138_复制带随机指针的链表)


- [q206_反转链表](/src/链表操作/q206_反转链表)

```java
class ListNode{
    int val;
    ListNode next;
    ListNode(int x){
        val = x;
    }
}
//Solution0 递归法
/*
总体来说，递归法是从最后一个Node开始，在弹栈的过程中将指针顺序置换的。
递归实质上就是系统帮你压栈的过程，系统在压栈的时候会保留现场。
我们来看是怎样的一个递归过程：0->2->3->4
程序到达Node newHead = reverse(head.next);时进入递归
我们假设此时递归到了2结点，此时head=3结点，temp=3结点.next(实际上是4结点)
执行Node newHead = reverse(head.next);传入的head.next是3结点，返回的newHead是4结点。
接下来就是弹栈过程了
程序继续执行 temp.next = head就相当于3->3
head.next = null 即把2结点指向4结点的指针断掉。
返回新链表的头结点newHead
注意：当retuen后，系统会恢复1结点压栈时的现场，此时的head=2结点；temp=2结点.next(3结点)，再进行上述的操作。最后完成整个链表的翻转。
 */
class Solution0{
    public ListNode reverseList(ListNode head){
        if(head == null || head.next == null)
            return head;

        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }
}

class Solution1{
    public ListNode reverse(ListNode head){
        if(head == null || head.next == null)
            return head;
        ListNode temp = head.next;
        ListNode newHead = reverse(head.next);
        temp.next = head;
        head.next = null;
        return newHead;
    }
}

//Solution1 遍历法
class Solution1{
    public ListNode reverseList(ListNode node){
        ListNode pre = null;
        ListNode next = null;
        while(node != null){
            next = node.next;
            node.next = pre;
            pre = node;
            node = next;
        }
        return pre;
    }
}
```

# 双指针遍历/滑动窗口

- [q3_无重复字符的最长子串](/src/双指针遍历/q3_无重复字符的最长子串)


- [q11_盛最多水的容器](/src/双指针遍历/q11_盛最多水的容器)

- [q15_三数之和](/src/双指针遍历/q15_三数之和)


- [q16_最接近的三数之和](/src/双指针遍历/q16_最接近的三数之和)


- [q26_删除排序数组中的重复项](/src/双指针遍历/q26_删除排序数组中的重复项)


- [q42_接雨水](/src/双指针遍历/q42_接雨水)


- [q121_买卖股票的最佳时机](/src/双指针遍历/q121_买卖股票的最佳时机)


- [q209_长度最小的子数组](/src/双指针遍历/q209_长度最小的子数组)




# 快慢指针遍历

- [q141_环形链表](/src/快慢指针遍历/q141_环形链表)

- [q202_快乐数](/src/快慢指针遍历/q202_快乐数)

- [q876_链表的中间结点](/src/快慢指针遍历/q876_链表的中间结点)





# 区间合并

- [q56_合并区间](/src/区间合并/q56_合并区间)

# 字符串操作

- [q6_Z字形变换](/src/字符串操作/q6_Z字形变换)



- [q14_最长公共前缀](/src/字符串操作/q14_最长公共前缀)



- [q763_划分字母区间](/src/字符串操作/q763_划分字母区间)

### 1.无重复字符的最长子串

**题目描述：**

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1:**

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

**Solution:**

```java
//这道题可以写几个例子试试就更好理解了。
public class Solution {
    public int lengthOfLongestSubstring(String s) {
        int[] map = new int[256];
        int l = 0, r = 0;
        int res = 0;
        while (l <= r && l < s.length()) {
            if (r < s.length() && map[s.charAt(r)] == 0) {
                map[s.charAt(r++)]++;
                res = Math.max(res, r - l);
            } else {
                map[s.charAt(l++)]--;
            }
        }
        return res;
    }
}
```



# 数字操作

- [q7_整数反转](/src/数字操作/q7_整数反转)


- [q8_字符串转换整数](/src/数字操作/q8_字符串转换整数)


- [q9_回文数](/src/数字操作/q9_回文数)


- [q43_字符串相乘](/src/数字操作/q43_字符串相乘)


- [q172_阶乘后的零](/src/数字操作/q172_阶乘后的零)


- [q258_各位相加](/src/数字操作/q258_各位相加)

# 数组操作

- [q54_螺旋矩阵](/src/数组操作/q54_螺旋矩阵)


- [q73_矩阵置零](/src/数组操作/q73_矩阵置零)


- [q78_子集](/src/数组操作/q78_子集)


- [q384_打乱数组](/src/数组操作/q384_打乱数组)


- [q581_最短无序连续子数组](/src/数组操作/q581_最短无序连续子数组)


- [q945_使数组唯一的最小增量](/src/数组操作/q945_使数组唯一的最小增量)



# 栈相关

- [q20_有效的括号](/src/栈相关/q20_有效的括号)


- [q32_最长有效括号](/src/栈相关/q32_最长有效括号)


- [q155_最小栈](/src/栈相关/q155_最小栈)


- [q224_基本计算器](/src/栈相关/q224_基本计算器)


- [q232_用栈实现队列](/src/栈相关/q232_用栈实现队列)


- [q316_去除重复字母](/src/栈相关/q316_去除重复字母)

# 堆相关

- [q215_数组中的第K个最大元素](/src/堆相关/q215_数组中的第K个最大元素)


- [q347_前K个高频元素](/src/堆相关/q347_前K个高频元素)


# 递归

- [q21_合并两个有序链表](/src/递归/q21_合并两个有序链表)


- [q101_对称二叉树](/src/递归/q101_对称二叉树)


- [q104_二叉树的最大深度](/src/递归/q104_二叉树的最大深度)


- [q226_翻转二叉树](/src/递归/q226_翻转二叉树)


- [q236_二叉树的最近公共祖先](/src/递归/q236_二叉树的最近公共祖先)


- [q1325_删除给定值的叶子节点](/src/递归/q1325_删除给定值的叶子节点)



# 分治法/二分法

- [q23_合并K个排序链表](/src/分治法/q23_合并K个排序链表)


- [q33_搜索旋转排序数组](/src/分治法/q33_搜索旋转排序数组)


- [q34_在排序数组中查找元素的第一个和最后一个位置](/src/分治法/q34_在排序数组中查找元素的第一个和最后一个位置)

### 1.寻找两个有序数组的中位数

**题目描述：**

给定两个大小为 m 和 n 的有序数组 `nums1` 和 `nums2`。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 `nums1` 和 `nums2` 不会同时为空。

**示例 1:**

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
```

**示例 2:**

```
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```

**Solution：**
方法1：更好理解
<https://www.nowcoder.com/discuss/196951>
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int l = (m + n + 1) / 2;
        int r = (m + n + 2) / 2;
        return (getKth(nums1, 0, nums2, 0, l) + getKth(nums1, 0, nums2, 0, r)) / 2.0;
    }

    // 在两个有序数组中二分查找第k大元素
    private int getKth(int[] nums1, int start1, int[] nums2, int start2, int k){
        // 特殊情况(1)，分析见正文部分
        if(start1 > nums1.length-1) return nums2[start2 + k - 1];
        if(start2 > nums2.length-1) return nums1[start1 + k - 1];
        // 特征情况(2)，分析见正文部分
        if(k == 1) return Math.min(nums1[start1], nums2[start2]);

        // 分别在两个数组中查找第k/2个元素，若存在（即数组没有越界），标记为找到的值；若不存在，标记为整数最大值
        int nums1Mid = start1 + k/2 - 1 < nums1.length ? nums1[start1 + k/2 - 1] : Integer.MAX_VALUE;
        int nums2Mid = start2 + k/2 - 1 < nums2.length ? nums2[start2 + k/2 - 1] : Integer.MAX_VALUE;

        // 确定最终的第k/2个元素，然后递归查找
        if(nums1Mid < nums2Mid)
            return getKth(nums1, start1 + k/2, nums2, start2, k-k/2);
        else
            return getKth(nums1, start1, nums2, start2 + k/2, k-k/2);
    }
}


方法2：
public class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int l = nums1.length + nums2.length + 1 >> 1;
        int r = nums1.length + nums2.length + 2 >> 1;
        return (find(nums1, nums2, 0, 0, l) + find(nums1, nums2, 0, 0, r)) / 2.0;
    }
    
    private double find(int[] nums1, int[] nums2, int start1, int start2, int k) {
        if (start1 >= nums1.length) {
            return nums2[start2 + k - 1];
        }
        if (start2 >= nums2.length) {
            return nums1[start1 + k - 1];
        }
        
        if (k == 1) {
            return Math.min(nums1[start1], nums2[start2]);
        }
        
        int mid1 = start1 + k / 2 - 1 >= nums1.length ? Integer.MAX_VALUE : nums1[start1 + k / 2 - 1];
        int mid2 = start2 + k / 2 - 1 >= nums2.length ? Integer.MAX_VALUE : nums2[start2 + k / 2 - 1];
        
        return mid1 > mid2 ? find(nums1, nums2, start1, start2 + k / 2, k - k / 2) :
            find(nums1, nums2, start1 + k / 2, start2, k - k / 2);
    }
}
```

# 动态规划

- [q5_最长回文子串](/src/动态规划/q5_最长回文子串)


- [q53_最大子序和](/src/动态规划/q53_最大子序和)


- [q62_不同路径](/src/动态规划/q62_不同路径)


- [q64_最小路径和](/src/动态规划/q64_最小路径和)


- [q70_爬楼梯](/src/动态规划/q70_爬楼梯)


- [q118_杨辉三角](/src/动态规划/q118_杨辉三角)


- [q300_最长上升子序列](/src/动态规划/q300_最长上升子序列)


- [q1143_最长公共子序列](/src/动态规划/q1143_最长公共子序列)


- [q1277_统计全为1的正方形子矩阵](/src/动态规划/q1277_统计全为1的正方形子矩阵)

# 回溯法

- [q10_正则表达式匹配](/src/回溯法/q10_正则表达式匹配)


- [q22_括号生成](/src/回溯法/q22_括号生成)


- [q40_组合总和2](/src/回溯法/q40_组合总和2)


- [q46_全排列](/src/回溯法/q46_全排列)

# 字典树（前缀树）

- [q648_单词替换](/src/字典树/q648_单词替换)

# 树的遍历

- [q94_二叉树的中序遍历](/src/树的遍历/q94_二叉树的中序遍历)


- [q102_二叉树的层次遍历](/src/树的遍历/q102_二叉树的层次遍历)


- [q110_平衡二叉树](/src/树的遍历/q110_平衡二叉树)


- [q144_二叉树的前序遍历](/src/树的遍历/q144_二叉树的前序遍历)


- [q145_二叉树的后序遍历](/src/树的遍历/q145_二叉树的后序遍历)



#  二叉搜索树相关

- [q98_验证二叉搜索树](/src/二叉搜索树相关/q98_验证二叉搜索树)


- [q450_删除二叉搜索树中的节点](/src/二叉搜索树相关/q450_删除二叉搜索树中的节点)


- [q701_二叉搜索树中的插入操作](/src/二叉搜索树相关/q701_二叉搜索树中的插入操作)

-------

## 面试问题整理

- [面试问题整理](/Rocket.md)

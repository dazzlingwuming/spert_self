from py2neo import Graph

if __name__ == '__main__':
    '''
    Graph接受三个参数
    profile是一个端口地址，也就是neo4j的bolt地址
    name和password是认证数据库的
    **settings是其他的配置参数
    具体可以参考官方文档：https://py2neo.org/v4/graph.html
    '''
    graph = Graph("neo4j://127.0.0.1:7687",name="neo4j" , auth=("neo4j", "12345678"))
    # a = graph.run("CREATE ( n :类型 {name: '无厘头'}) return n")
    # print(a)
    b = graph.run("match (n:类型 {name: '无厘头'}) set n.movies = $movies return n",movies = "唐伯虎点秋香").data()
    '''
    使用参数化查询可以防止cypher注入攻击，直接在查询语句中使用变量占位符（如 $movies），然后通过关键字参数传递实际的值。
    '''
    print(b)
    r = graph.run("MATCH (n:类型) RETURN n").data()
    '''
    .evaluate()#evaluate() 的作用：返回查询结果的第一行第一列的值（如果没有结果则返回 None）。也就是说，当你的 Cypher 返回一个字段（比如节点 n）时，evaluate() 会直接给出第一个匹配的那个值；若需要所有行或所有列请使用 data()/ to_table() 等方法。
    .data()#data() 的作用：将查询结果转换为一个列表，列表中的每个元素都是一个字典，字典的键是查询中返回的字段名，值是对应的值。也就是说，data() 会把所有匹配的结果都返回出来，适合需要处理多行数据的情况。
    '''
    print(r)

    '''
    graph.run() 方法用于执行 Cypher 查询语句，并返回一个 Cursor 对象。这个对象可以用来遍历查询结果，或者通过调用 evaluate()、data() 等方法来获取结果数据。
    '''
    record = graph.run("MATCH (n:类型) RETURN n limit 5")
    while record.forward():
        print("---"*10)
        current = record.current
        print(current)#获取当前记录
        print(current["n"]["类型ID"])#通过字段名获取当前记录的值
        print(current["n"].start_node)#获取节点的起始节点
        print(current["n"].end_node)#获取节点的结束节点
        print(current["n"].relationships)#获取节点的关系

from neo4j import GraphDatabase

class HelloWorldExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def print_greeting(self, message):
        with self.driver.session() as session:
            greeting = session.execute_write(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

def t1():
    with GraphDatabase.driver("bolt://localhost:7687") as driver:
        driver.verify_connectivity()
        '''
        1.非事务性操作：适合执行简单的读写操作，不需要复杂的事务管理：
            record, summary , key = driver.execute_query("MATCH (n:'类型‘ {类型名称 ：$name}) RETURN n LIMIT 1" , name="动作")
            record, summary , key 分别是查询结果、查询摘要和返回字段名列表。
        2.事务性操作：适合需要多步操作的复杂事务，可以确保:
            with self.driver.session(database = "neo4j") as session:
                greeting = session.execute_write(self._create_and_return_greeting, message)
                print(greeting)
        '''

if __name__ == "__main__":
    greeter = HelloWorldExample("bolt://:7687", "neo4j", "password")
    greeter.print_greeting("hello, world")
    greeter.close()
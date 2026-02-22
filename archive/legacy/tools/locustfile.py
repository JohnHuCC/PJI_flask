from locust import HttpUser, task, between


class WebsiteUser(HttpUser):
    host = "https://pji.nycu.page"
    wait_time = between(5, 15)

    def on_start(self):
        """ 在任何任務開始之前執行一次的方法，用於登錄 """
        self.client.post("https://pji.nycu.page/auth_login", {
            "username": "test",
            "password": "1234"
        })

    @task(1)
    def visit_page_after_login(self):
        """ 登錄後訪問的頁面 """
        self.client.get("https://pji.nycu.page/personal_info?p_id=62")

    @task(2)
    def model_diagnosis(self):
        """ 訪問另一個頁面的任務 """
        self.client.get("https://pji.nycu.page/model_diagnosis?p_id=62")

    @task(3)
    def reactive_diagram(self):
        """ 訪問另一個頁面的任務 """
        self.client.get(
            "https://pji.nycu.page/reactive_diagram?p_id=62&signal=1")

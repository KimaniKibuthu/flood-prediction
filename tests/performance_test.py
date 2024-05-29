from locust import HttpUser, TaskSet, task, between


class UserBehavior(TaskSet):
    @task
    def predict(self):
        self.client.post("/v1/flood-prediction", json={"location": "Dhaka"})


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)

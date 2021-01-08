from locust import HttpUser, task, between

class WebsiteTestUser(HttpUser):
    wait_time = between(0.5, 3.0)

    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        pass

    def on_stop(self):
        """ on_stop is called when the TaskSet is stopping """
        pass

#    @task
#    def home(self):
#        self.client.get("http://localhost:5000/")

#    @task
#    def docs(self):
#        self.client.get("http://localhost:5000/apidocs/")

#    @task
#    def godzilla(self):
#        self.client.get("http://localhost:5000/api/intent/Godzilla 2/")

    @task
    def avion(self):
        self.client.get("api/intent/Je veux réserver un avion/")
    #@task
    #def supermarche(self):
    #    self.client.get("http://localhost:5000/api/intent/Existe-t-il un supermarché près de moi ?/")

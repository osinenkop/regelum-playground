import regelum

from regelum import callback

class StateLogger(callback.StateTracker):
    def on_trigger(self, caller):
        self.log(f"STATE: {self.system_state}")

class ObservationLogger(callback.ObservationTracker):
    def on_trigger(self, caller):
        self.log(f"OBSERVATION: {self.observation}")

class MomentOfInertiaLogger(callback.Callback):

    def is_target_event(self, obj, method, output, triggers):
        return method == "pendulum_moment_inertia"

    def on_function_call(self, obj, method, output):
        self.log(f"Moment of inertia: {output}")
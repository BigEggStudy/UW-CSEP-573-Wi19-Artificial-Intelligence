"""

Loading POMDP environment files.

author: mbforbes
some additions: koosha

"""

import numpy as np


class POMDP:
    def __init__(self, filename):
        """
        Parses .pomdp file and loads info into this object's fields.
        Attributes:
            discount
            values
            states
            actions
            observations
            T(action, start_state, next_state)
            Z(action, next_state, observation)
            R(action, start_state, next_state, observation)
            prior 
        """
        f = open(filename, 'r')
        self.contents = [
            x.strip() for x in f.readlines()
            if (not (x.startswith("#") or x.isspace()))
        ]


        # go through line by line
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            if line.startswith('discount'):
                i = self.__get_discount(i)
            elif line.startswith('values'):
                i = self.__get_value(i)
            elif line.startswith('states'):
                i = self.__get_states(i)
            elif line.startswith('actions'):
                i = self.__get_actions(i)
            elif line.startswith('observations'):
                i = self.__get_observations(i)
            elif line.startswith('start'):
                i = self.__get_prior(i)
                self.T = np.zeros([len(self.actions), len(self.states), len(self.states)])
                self.O = np.zeros([len(self.actions), len(self.states), len(self.observations)])
                self.R = np.zeros([len(self.actions), len(self.states), len(self.states), len(self.observations)])
            elif line.startswith('T'):
                i = self.__get_transition(i)
            elif line.startswith('O'):
                i = self.__get_observation(i)
            elif line.startswith('R'):
                i = self.__get_reward(i)
            else:
                raise Exception("Unrecognized line: " + line)

        f.close()
#        

    def __get_discount(self, i):
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_value(self, i):
        # Currently just supports "values: reward". I.e. currently
        # meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __get_states(self, i):
        line = self.contents[i]
        self.states = line.split()[1:]
        if len(self.states) == 1 and int(self.states[0])>0:
            self.states = [str(x) for x in range(0, int(self.states[0]))]
        return i + 1

    def __get_actions(self, i):
        line = self.contents[i]
        self.actions = line.split()[1:]
        if len(self.actions) == 1 and int(self.actions[0])>0:
            self.actions = [str(x) for x in range(0, int(self.actions[0]))]
        return i + 1

    def __get_observations(self, i):
        line = self.contents[i]
        self.observations = line.split()[1:]
        if len(self.observations) == 1 and int(self.observations[0])>0:
            self.observations = [str(x) for x in range(0, int(self.observations[0]))]
        return i + 1
    
    def __get_prior(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        self.prior = np.array([float(x) for x in pieces])
        return i + 1
    
    def __get_transition(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = -1 
        if pieces[0] != "*":
            action = self.actions.index(pieces[0])
            
        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            prob = float(pieces[3])
            if action > -1:    
                self.T[action, start_state, next_state] = prob
            else:
                self.T[:, start_state, next_state] = prob
                
            return i + 1
        
        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            if action > -1:    
                self.T[action, start_state, next_state] = prob
            else:
                self.T[:, start_state, next_state] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.states)
            for j in range(len(probs)):
                prob = float(probs[j])
                if action > -1:    
                    self.T[action, start_state, j] = prob
                else:
                    self.T[:, start_state, j] = prob
                    action = -1
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: T: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        prob = 1.0 if j == k else 0.0
                        if action > -1:    
                            self.T[action, j, k] = prob
                        else:
                            self.T[:, j, k] = prob
        
                return i + 2
            elif next_line == "uniform":
                # case 5: T: <action>
                # uniform
                prob = 1.0 / float(len(self.states))
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        self.T[action, j, k] = prob
                return i + 2
            else:
                # case 6: T: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.states)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.T[action, j, k] = prob
                    next_line = self.contents[i+2+j]
                return i+1+len(self.states)
        else:
            raise Exception("Cannot parse line " + line)

    def __get_observation(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = -1 
        if pieces[0] != "*":
            action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            next_state = self.states.index(pieces[1])
            obs = self.observations.index(pieces[2])
            prob = float(pieces[3])
            if action > -1:    
                self.O[action, next_state, obs] = prob
            else:
                self.O[:, next_state, obs] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: O: <action> : <next-state> : <obs>
            # %f
            next_state = self.states.index(pieces[1])
            obs = self.observations.index(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.O[action, next_state, obs] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            next_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                if action > -1:
                    self.O[action, next_state, j] = prob
                else:
                    self.O[:, next_state, j] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: O: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        prob = 1.0 if j == k else 0.0
                        self.O[action, j, k] = prob
                return i + 2
            elif next_line == "uniform":
                # case 5: O: <action>
                # uniform
                prob = 1.0 / float(len(self.observations))
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        self.O[action, j, k] = prob
                return i + 2
            else:
                # case 6: O: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.observations)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.O[action, j, k] = prob
                    next_line = self.contents[i+2+j]
                return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __get_reward(self, i):
        """
        Wild card * are allowed when specifying a single reward
        probability. They are not allowed when specifying a vector or
        matrix of probabilities, but we handeled them for non-standard pomdp models.
        """
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]

        if len(pieces) == 5 or len(pieces) == 4:
            # case 1:
            # R: <action> : <start-state> : <next-state> : <obs> %f
            # any of <start-state>, <next-state>, and <obs> can be *
            # %f can be on the next line (case where len(pieces) == 4)
            action_raw = pieces[0]
            start_state_raw = pieces[1]
            next_state_raw = pieces[2]
            obs_raw = pieces[3]
            prob = float(pieces[4]) if len(pieces) == 5 \
                else float(self.contents[i+1])
            self.__reward_a(
                action_raw, start_state_raw, next_state_raw, obs_raw, prob)
            return i + 1 if len(pieces) == 5 else i + 2
        elif len(pieces == 3):
            # case 2: R: <action> : <start-state> : <next-state>
            # %f %f ... %f
            action = self.actions.index(pieces[0])
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.R[action, start_state, next_state, j] = prob
            return i + 2
        elif len(pieces == 2):
            # case 3: R: <action> : <start-state>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.R[action, start_state, j, k] = prob
                next_line = self.contents[i+2+j]
            return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __reward_a(self, action_raw, start_state_raw, next_state_raw, obs_raw, prob):
        """
        reward_a means we're at the action of the unrolling of the
        reward expression. action could be * or the name of the
        real start state.
        """
        if action_raw == '*':
            for a in range(len(self.actions)):
                self.__reward_ss(a, start_state_raw, next_state_raw, obs_raw, prob)
        else:
           action = self.actions.index(action_raw)
           self.__reward_ss(action, start_state_raw, next_state_raw, obs_raw, prob)
            
    def __reward_ss(self, a, start_state_raw, next_state_raw, obs_raw, prob):
        """
        reward_ss means we're at the start state of the unrolling of the
        reward expression. start_state_raw could be * or the name of the
        real start state.
        """
        if start_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ns(a, i, next_state_raw, obs_raw, prob)
        else:
            start_state = self.states.index(start_state_raw)
            self.__reward_ns(a, start_state, next_state_raw, obs_raw, prob)

    def __reward_ns(self, a, start_state, next_state_raw, obs_raw, prob):
        """
        reward_ns means we're at the next state of the unrolling of the
        reward expression. start_state is the number of the real start
        state, and next_state_raw could be * or the name of the real
        next state.
        """
        if next_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ob(a, start_state, i, obs_raw, prob)
        else:
            next_state = self.states.index(next_state_raw)
            self.__reward_ob(a, start_state, next_state, obs_raw, prob)

    def __reward_ob(self, a, start_state, next_state, obs_raw, prob):
        """
        reward_ob means we're at the observation of the unrolling of the
        reward expression. start_state is the number of the real start
        state, next_state is the number of the real next state, and
        obs_raw could be * or the name of the real observation.
        """
        if obs_raw == '*':
            self.R[a, start_state, next_state, :] = prob
        else:
            obs = self.observations.index(obs_raw)
            self.R[a, start_state, next_state, obs] = prob

    def print_summary(self):
        print ("discount:", self.discount)
        print ("values:", self.values)
        print ("states:", self.states)
        print ("actions:", self.actions)
        print ("observations:", self.observations)
        print ("")
        print ("T:", self.T)
        print ("")
        print ("O:", self.O)
        print ("")
        print ("R:", self.R)




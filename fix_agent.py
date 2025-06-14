#\!/usr/bin/env python3

# Read the agent.py file
with open('src/active_inference/agent.py', 'r') as f:
    content = f.read()

# Add the get_current_beliefs method after the __init__ method
init_end = 'logger.info(f"ActiveInferenceAgent initialized with pymdp={self.use_pymdp}")'
new_method = '''
    
    def get_current_beliefs(self) -> BeliefState:
        """Get the current belief state."""
        if self.belief_state is None:
            logger.warning("No belief state initialized, returning empty state")
            return self._create_empty_belief_state()
        return self.belief_state'''

# Insert the new method
content = content.replace(init_end, init_end + new_method)

# Write back to file
with open('src/active_inference/agent.py', 'w') as f:
    f.write(content)

print("Added get_current_beliefs method to agent.py")

# FedDefender: Backdoor Attack Defense in Federated Learning (Tutorial)

`This tutorial is based on a paper accepted at SE4SafeML: Dependability and Trustworthiness of Safety-Critical Systems with Machine Learned Components (Colocated with FSE 2023). The ArXiv version of the manuscript is available` [here](https://arxiv.org/abs/2307.08672).

Federated Learning is a method for training machine learning models across multiple devices or servers, each holding local data samples, without the need to exchange them. This approach is excellent for privacy, but it also introduces the possibility of certain types of attacks. FedDefender is designed to address this issue.

## The Problem: Poisoning Attacks

Consider a group project where everyone contributes a part. If one person starts providing incorrect or misleading information, it could disrupt the entire project. In Federated Learning, this is known as a poisoning attack. An attacker can manipulate the learning process by injecting false data from a malicious client.

## The Solution: FedDefender

FedDefender is a novel defense mechanism designed to safeguard Federated Learning from these poisoning attacks. Instead of attempting to prevent attacks, it focuses on identifying and mitigating them. It accomplishes this by leveraging a technique known as differential testing.

### Superpower: Differential Testing

FedDefender's superpower is "differential testing." This concept might sound complex, but it's actually quite straightforward.

Imagine being a teacher and giving all your students the same question to answer. If one student provides a significantly different answer from everyone else, you might suspect that something is amiss. Maybe they didn't understand the question, or perhaps they're trying to be humorous. Either way, you know you need to investigate what's happening.

This is precisely what FedDefender does. It provides the same piece of information to all the devices (or "students") and then examines how each device's model (or "answer") responds. If one model reacts very differently from the others, FedDefender knows something might be amiss.

### How Does FedDefender Work? Neuron Activations as Fingerprints

In differential testing, FedDefender compares the neuron activations of clients' models on the same input. By comparing these "fingerprints," it can identify clients that are behaving differently from the rest - a sign that they may be malicious.

Once a potentially malicious client is identified, FedDefender can take action to mitigate the threat. This could involve excluding the client from the learning process or flagging it for further investigation.

## Why is FedDefender Important?

FedDefender has proven to be highly effective in defending against backdoor attacks in Federated Learning. In evaluations, it reduced the attack success rate to just 10%, without significantly affecting the performance of the global model. This makes it a powerful tool for any organization using Federated Learning.

## Conclusion

As Federated Learning continues to grow in popularity, so too will the threats against it. Defense mechanisms like FedDefender will be crucial in ensuring the security and integrity of this promising approach to machine learning. By identifying and mitigating attacks, rather than just trying to prevent them, FedDefender represents a significant step forward in the fight against poisoning attacks.

---

Remember, this is a simplified explanation of a complex topic. For a more detailed understanding, I would recommend reading the original paper.

For any questions regarding FedDefender's artifact, please direct them to Waris Gill at waris@vt.edu.

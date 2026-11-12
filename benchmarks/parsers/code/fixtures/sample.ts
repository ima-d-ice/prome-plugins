interface UserProfile {
    id: number;
    username: string;
    isActive: boolean;
}

class UserService {
    private users: UserProfile[] = [];

    findUser(id: number): UserProfile | undefined {
        return this.users.find(u => u.id === id);
    }
}

function formatGreeting(user: UserProfile): string {
    return `Hello, ${user.username}!`;
}
import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  import.meta.env.PUBLIC_SUPABASE_URL,
  import.meta.env.PUBLIC_SUPABASE_ANON_KEY
);

export interface Message {
  id: string;
  nickname: string;
  content: string;
  parent_id: string | null;
  is_admin: boolean;
  created_at: string;
  replies?: Message[];
}
